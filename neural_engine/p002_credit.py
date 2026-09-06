from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CreditUpdate:
    circuit_id: int
    target_step: int
    target_slot: int
    advantage: float
    eligible_examples: int
    auxiliary_loss: float
    down_grad: torch.Tensor
    up_grad: torch.Tensor
    bias_grad: torch.Tensor


class CounterfactualCircuitCredit:
    """Training-only sparse credit for starved independent micro-circuits.

    The live router is never changed. At a configurable interval, a small set
    of under-trained circuit rows are evaluated by replaying the *same* route,
    weights and route gains while replacing one route slot. The best shadow
    candidate receives task-loss gradient only on its own circuit row.

    This deliberately separates two questions:
      1. can a starved circuit become useful if it receives task-aligned credit?
      2. can the existing router later discover that useful circuit?

    Only (1) is changed by this experiment; normal on-policy loss continues to
    train the original model exactly as before.
    """

    def __init__(
        self,
        num_circuits: int,
        internal_steps: int,
        active_circuits: int,
        *,
        interval: int = 8,
        candidates: int = 4,
        weight: float = 0.25,
        min_eligible: int = 16,
        seed: int = 0,
    ) -> None:
        if num_circuits < 2:
            raise ValueError("counterfactual credit requires at least two circuits")
        if internal_steps < 1 or active_circuits < 1:
            raise ValueError("internal_steps and active_circuits must be positive")
        if interval < 1 or candidates < 1:
            raise ValueError("interval and candidates must be positive")
        if weight <= 0:
            raise ValueError("weight must be positive")
        if min_eligible < 1:
            raise ValueError("min_eligible must be positive")
        self.num_circuits = int(num_circuits)
        self.internal_steps = int(internal_steps)
        self.active_circuits = int(active_circuits)
        self.interval = int(interval)
        self.candidates = min(int(candidates), self.num_circuits)
        self.weight = float(weight)
        self.min_eligible = int(min_eligible)
        self.rng = random.Random(int(seed))

        self.on_policy_usage = torch.zeros(self.num_circuits, dtype=torch.long)
        self.on_policy_forward_examples = torch.zeros(self.num_circuits, dtype=torch.long)
        self.main_grad_sample_count = torch.zeros(self.num_circuits, dtype=torch.long)
        self.main_grad_nonzero_samples = torch.zeros(self.num_circuits, dtype=torch.long)
        self.main_grad_norm_sum = torch.zeros(self.num_circuits, dtype=torch.float64)
        self.shadow_probe_examples = torch.zeros(self.num_circuits, dtype=torch.long)
        self.shadow_update_events = torch.zeros(self.num_circuits, dtype=torch.long)
        self.shadow_update_examples = torch.zeros(self.num_circuits, dtype=torch.long)
        self.shadow_advantage_sum = torch.zeros(self.num_circuits, dtype=torch.float64)
        self.shadow_aux_grad_norm_sum = torch.zeros(self.num_circuits, dtype=torch.float64)
        self.events_attempted = 0
        self.events_applied = 0

    @property
    def training_forward_overhead(self) -> float:
        """Planned extra full forwards per normal training forward."""
        return (self.candidates + 1) / self.interval

    @property
    def training_aux_backward_overhead(self) -> float:
        """Planned extra circuit-only backwards per normal training step."""
        return 1.0 / self.interval

    def observe_on_policy(self, selected_ids: torch.Tensor) -> None:
        selected = selected_ids.detach().to("cpu").reshape(-1)
        selected = selected[selected.ge(0)]
        if selected.numel() == 0:
            return
        self.on_policy_usage += torch.bincount(selected, minlength=self.num_circuits)

        # Count examples touching each circuit, rather than only route slots.
        per_example = selected_ids.detach().to("cpu").reshape(selected_ids.shape[0], -1)
        for circuit_id in range(self.num_circuits):
            self.on_policy_forward_examples[circuit_id] += int(
                per_example.eq(circuit_id).any(dim=1).sum()
            )

    def observe_main_grad(self, circuits: nn.Module, circuit_id: int) -> None:
        """Sample one row's main-loss gradient without scanning the whole bank."""
        circuit_id = int(circuit_id)
        squared = None
        for name in ("down", "up", "bias"):
            parameter = getattr(circuits, name, None)
            if parameter is None or parameter.grad is None:
                continue
            value = parameter.grad[circuit_id].detach().float().pow(2).sum()
            squared = value if squared is None else squared + value
        if squared is None:
            return
        norm = float(squared.sqrt().cpu())
        self.main_grad_sample_count[circuit_id] += 1
        self.main_grad_norm_sum[circuit_id] += norm
        if norm > 1e-12:
            self.main_grad_nonzero_samples[circuit_id] += 1

    def _candidate_ids(self) -> list[int]:
        # Prefer circuits that have received the fewest shadow updates and the
        # least on-policy use. Random tie-breaking is isolated from global RNG,
        # so the control arm sees the same model/data randomness.
        tie = list(range(self.num_circuits))
        self.rng.shuffle(tie)
        tie_rank = {circuit_id: rank for rank, circuit_id in enumerate(tie)}
        ranked = sorted(
            range(self.num_circuits),
            key=lambda circuit_id: (
                int(self.shadow_update_events[circuit_id]),
                int(self.on_policy_usage[circuit_id]),
                tie_rank[circuit_id],
            ),
        )
        return ranked[: self.candidates]

    def _target(self, step_number: int) -> tuple[int, int]:
        event_index = max(0, step_number // self.interval - 1)
        flat = event_index % (self.internal_steps * self.active_circuits)
        return flat // self.active_circuits, flat % self.active_circuits

    def should_run(self, step_number: int) -> bool:
        return step_number > 0 and step_number % self.interval == 0

    def prepare_update(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor,
        route_stats: dict[str, torch.Tensor],
        step_number: int,
    ) -> CreditUpdate | None:
        if not self.should_run(step_number):
            return None
        self.events_attempted += 1

        circuits = getattr(model, "circuits", None)
        if circuits is None or not all(hasattr(circuits, name) for name in ("down", "up", "bias")):
            raise TypeError("P-002 credit experiment requires independent MicroCircuitBank rows")

        selected = route_stats["selected_ids"].detach()
        weights = route_stats["selected_weights"].detach()
        gains = route_stats["route_gains"].detach()
        expected = (inputs.shape[0], self.internal_steps, self.active_circuits)
        if tuple(selected.shape) != expected:
            raise ValueError(
                f"P-002 expects hard routes with shape {expected}, got {tuple(selected.shape)}"
            )
        if tuple(weights.shape) != expected:
            raise ValueError("P-002 requires selected_weights matching hard selected_ids")

        baseline_loss = F.cross_entropy(logits.detach(), targets, reduction="none")
        target_step, target_slot = self._target(step_number)
        candidates = self._candidate_ids()
        scores: list[tuple[float, int, torch.Tensor]] = []

        for circuit_id in candidates:
            # A candidate is eligible only where it is absent from the whole
            # on-policy trajectory. This makes the later row-masked gradient a
            # pure counterfactual update for that circuit.
            already_used = selected.eq(circuit_id).reshape(selected.shape[0], -1).any(dim=1)
            executed = selected[:, target_step, target_slot].ge(0)
            eligible = (~already_used) & executed
            eligible_count = int(eligible.sum())
            if eligible_count < self.min_eligible:
                continue

            forced = selected.clone()
            forced[eligible, target_step, target_slot] = circuit_id
            with torch.no_grad():
                candidate_logits, _ = model(
                    inputs,
                    adaptive=False,
                    forced_selected_ids=forced,
                    forced_selected_weights=weights,
                    forced_route_gains=gains,
                )
                candidate_loss = F.cross_entropy(candidate_logits, targets, reduction="none")
                advantage = float((baseline_loss[eligible] - candidate_loss[eligible]).mean().cpu())
            self.shadow_probe_examples[circuit_id] += eligible_count
            scores.append((advantage, circuit_id, eligible))

        if not scores:
            return None

        advantage, winner, eligible = max(scores, key=lambda item: item[0])
        forced = selected.clone()
        forced[eligible, target_step, target_slot] = winner
        winner_logits, _ = model(
            inputs,
            adaptive=False,
            forced_selected_ids=forced,
            forced_selected_weights=weights,
            forced_route_gains=gains,
        )
        raw_aux_loss = F.cross_entropy(winner_logits[eligible], targets[eligible])
        aux_loss = self.weight * raw_aux_loss
        parameters = (circuits.down, circuits.up, circuits.bias)
        gradients = torch.autograd.grad(aux_loss, parameters, allow_unused=False)

        # Store only the winning row. All other circuit/controller/router/output
        # gradients from the counterfactual graph are intentionally discarded.
        row_grads = [gradient[winner].detach().clone() for gradient in gradients]
        eligible_count = int(eligible.sum())
        return CreditUpdate(
            circuit_id=int(winner),
            target_step=int(target_step),
            target_slot=int(target_slot),
            advantage=float(advantage),
            eligible_examples=eligible_count,
            auxiliary_loss=float(aux_loss.detach().cpu()),
            down_grad=row_grads[0],
            up_grad=row_grads[1],
            bias_grad=row_grads[2],
        )

    def apply_update(self, circuits: nn.Module, update: CreditUpdate | None) -> None:
        if update is None:
            return
        winner = update.circuit_id
        for name, row_grad in (
            ("down", update.down_grad),
            ("up", update.up_grad),
            ("bias", update.bias_grad),
        ):
            parameter = getattr(circuits, name)
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad[winner].add_(row_grad.to(parameter.grad.device, parameter.grad.dtype))

        aux_norm = torch.stack([
            update.down_grad.float().pow(2).sum(),
            update.up_grad.float().pow(2).sum(),
            update.bias_grad.float().pow(2).sum(),
        ]).sum().sqrt()
        self.shadow_update_events[winner] += 1
        self.shadow_update_examples[winner] += update.eligible_examples
        self.shadow_advantage_sum[winner] += update.advantage
        self.shadow_aux_grad_norm_sum[winner] += float(aux_norm.cpu())
        self.events_applied += 1

    def report(self) -> dict[str, Any]:
        usage = self.on_policy_usage
        grad_samples = self.main_grad_sample_count
        grad_nonzero = self.main_grad_nonzero_samples
        shadow_events = self.shadow_update_events
        used = usage.gt(0)
        trained = usage.gt(0) | shadow_events.gt(0)
        gradient_rate = grad_nonzero.float() / grad_samples.clamp_min(1).float()
        undertrained = grad_samples.gt(0) & gradient_rate.lt(0.25) & shadow_events.eq(0)

        circuits = []
        for circuit_id in range(self.num_circuits):
            events = int(shadow_events[circuit_id])
            circuits.append({
                "circuit_id": circuit_id,
                "on_policy_usage": int(usage[circuit_id]),
                "on_policy_forward_examples": int(self.on_policy_forward_examples[circuit_id]),
                "main_grad_sample_count": int(grad_samples[circuit_id]),
                "main_grad_nonzero_samples": int(grad_nonzero[circuit_id]),
                "main_grad_nonzero_rate": (
                    float(gradient_rate[circuit_id]) if grad_samples[circuit_id] else None
                ),
                "main_grad_norm_sum": float(self.main_grad_norm_sum[circuit_id]),
                "shadow_probe_examples": int(self.shadow_probe_examples[circuit_id]),
                "shadow_update_events": events,
                "shadow_update_examples": int(self.shadow_update_examples[circuit_id]),
                "shadow_mean_advantage": (
                    float(self.shadow_advantage_sum[circuit_id]) / events if events else None
                ),
                "shadow_aux_grad_norm_sum": float(self.shadow_aux_grad_norm_sum[circuit_id]),
            })

        return {
            "events_attempted": self.events_attempted,
            "events_applied": self.events_applied,
            "dead_on_policy_circuits": int((~used).sum()),
            "never_trained_circuits": int((~trained).sum()),
            "undertrained_circuits": int(undertrained.sum()),
            "planned_extra_forward_equivalents_per_step": self.training_forward_overhead,
            "planned_extra_circuit_backward_equivalents_per_step": self.training_aux_backward_overhead,
            "circuits": circuits,
        }
