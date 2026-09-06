from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CircuitRowCredit:
    circuit_id: int
    examples: int
    mean_advantage: float
    auxiliary_loss: float
    down_grad: torch.Tensor
    up_grad: torch.Tensor
    bias_grad: torch.Tensor


@dataclass
class CreditUpdate:
    target_step: int
    target_slot: int
    rows: list[CircuitRowCredit]
    responsible_examples: int
    mean_advantage: float


def _nmi_from_counts(counts: torch.Tensor) -> tuple[float, float, float]:
    """Return task/circuit NMI, weighted specialization and weighted purity."""
    matrix = counts.double()
    if matrix.numel() == 0 or float(matrix.sum()) <= 0:
        return 0.0, 0.0, 0.0
    circuit_load = matrix.sum(0)
    used = circuit_load.gt(0)
    probabilities = torch.zeros_like(matrix)
    probabilities[:, used] = matrix[:, used] / circuit_load[used].unsqueeze(0)
    entropy = torch.zeros(matrix.shape[1], dtype=torch.float64)
    purity = torch.zeros(matrix.shape[1], dtype=torch.float64)
    if used.any():
        p = probabilities[:, used]
        entropy[used] = -(p * p.clamp_min(1e-30).log()).sum(0) / math.log(matrix.shape[0])
        purity[used] = p.max(0).values
    weights = circuit_load / circuit_load.sum().clamp_min(1)
    specialization = float((weights * (1.0 - entropy)).sum())
    weighted_purity = float((weights * purity).sum())

    joint = matrix / matrix.sum().clamp_min(1)
    pt = joint.sum(1, keepdim=True)
    pc = joint.sum(0, keepdim=True)
    nz = joint.gt(0)
    ratio = joint / (pt * pc).clamp_min(1e-30)
    mi = float((joint[nz] * ratio[nz].log()).sum())
    ht = float(-(pt[pt.gt(0)] * pt[pt.gt(0)].log()).sum())
    hc = float(-(pc[pc.gt(0)] * pc[pc.gt(0)].log()).sum())
    nmi = mi / math.sqrt(max(ht * hc, 1e-30))
    return nmi, specialization, weighted_purity


class CounterfactualCircuitCredit:
    """Training-only causal responsibility for starved independent circuits.

    The live router and inference path are unchanged. At a configurable
    interval, a small set of low-usage circuit rows is evaluated by replaying
    the exact on-policy route while replacing one route slot. Credit is assigned
    *per example*: each example chooses the shadow circuit that most reduces the
    final task loss, and only examples above ``min_advantage`` train that row.

    This is intentionally different from coverage/exploration or a dense soft
    mixture. A row receives auxiliary gradient only for examples where it has
    measured positive final-loss responsibility. Controller/router/output
    gradients from the auxiliary replay are discarded.
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
        min_responsible: int = 4,
        min_advantage: float = 0.02,
        advantage_clip: float = 1.0,
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
        if min_eligible < 1 or min_responsible < 1:
            raise ValueError("min_eligible and min_responsible must be positive")
        if min_advantage < 0 or advantage_clip <= 0:
            raise ValueError("invalid advantage thresholds")
        if min_advantage >= advantage_clip:
            raise ValueError("min_advantage must be smaller than advantage_clip")
        self.num_circuits = int(num_circuits)
        self.internal_steps = int(internal_steps)
        self.active_circuits = int(active_circuits)
        self.interval = int(interval)
        self.candidates = min(int(candidates), self.num_circuits)
        self.weight = float(weight)
        self.min_eligible = int(min_eligible)
        self.min_responsible = int(min_responsible)
        self.min_advantage = float(min_advantage)
        self.advantage_clip = float(advantage_clip)
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
        self.responsibility_counts = torch.zeros(15, self.num_circuits, dtype=torch.long)
        self.events_attempted = 0
        self.events_applied = 0
        self.probe_forward_examples = 0
        self.gradient_forward_examples = 0
        self.responsible_examples = 0

    @property
    def planned_probe_forward_overhead(self) -> float:
        return self.candidates / self.interval

    @property
    def planned_aux_backward_upper_bound(self) -> float:
        return self.candidates / self.interval

    def observe_on_policy(self, selected_ids: torch.Tensor) -> None:
        selected = selected_ids.detach().to("cpu").reshape(-1)
        selected = selected[selected.ge(0)]
        if selected.numel() == 0:
            return
        self.on_policy_usage += torch.bincount(selected, minlength=self.num_circuits)

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
        # Starvation first. Shadow-update count is only a tie-breaker, so this
        # does not degenerate into round-robin coverage.
        tie = list(range(self.num_circuits))
        self.rng.shuffle(tie)
        tie_rank = {circuit_id: rank for rank, circuit_id in enumerate(tie)}
        ranked = sorted(
            range(self.num_circuits),
            key=lambda circuit_id: (
                int(self.on_policy_usage[circuit_id]),
                int(self.shadow_update_events[circuit_id]),
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
        task_ids: torch.Tensor | None = None,
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
        if task_ids is not None and task_ids.shape[0] != inputs.shape[0]:
            raise ValueError("task_ids must have one value per example")

        baseline_loss = F.cross_entropy(logits.detach(), targets, reduction="none")
        target_step, target_slot = self._target(step_number)
        candidates = self._candidate_ids()
        batch_size = inputs.shape[0]
        device = inputs.device
        advantage_matrix = torch.full(
            (len(candidates), batch_size), -torch.inf, device=device, dtype=baseline_loss.dtype
        )

        for candidate_index, circuit_id in enumerate(candidates):
            already_used = selected.eq(circuit_id).reshape(batch_size, -1).any(dim=1)
            executed = selected[:, target_step, target_slot].ge(0)
            eligible = (~already_used) & executed
            eligible_count = int(eligible.sum())
            if eligible_count < self.min_eligible:
                continue

            forced = selected[eligible].clone()
            forced[:, target_step, target_slot] = circuit_id
            with torch.no_grad():
                candidate_logits, _ = model(
                    inputs[eligible],
                    adaptive=False,
                    forced_selected_ids=forced,
                    forced_selected_weights=weights[eligible],
                    forced_route_gains=gains[eligible],
                )
                candidate_loss = F.cross_entropy(
                    candidate_logits, targets[eligible], reduction="none"
                )
                advantage_matrix[candidate_index, eligible] = (
                    baseline_loss[eligible] - candidate_loss
                )
            self.shadow_probe_examples[circuit_id] += eligible_count
            self.probe_forward_examples += eligible_count

        best_advantage, winner_index = advantage_matrix.max(dim=0)
        responsible = torch.isfinite(best_advantage) & best_advantage.ge(self.min_advantage)
        if not responsible.any():
            return None

        row_updates: list[CircuitRowCredit] = []
        total_advantage = 0.0
        total_examples = 0
        parameters = (circuits.down, circuits.up, circuits.bias)

        for candidate_index, circuit_id in enumerate(candidates):
            row_mask = responsible & winner_index.eq(candidate_index)
            support = int(row_mask.sum())
            if support < self.min_responsible:
                continue

            # The candidate was absent from every selected trajectory used for
            # this row, so extracting only its gradient gives pure shadow credit.
            forced = selected[row_mask].clone()
            forced[:, target_step, target_slot] = circuit_id
            winner_logits, _ = model(
                inputs[row_mask],
                adaptive=False,
                forced_selected_ids=forced,
                forced_selected_weights=weights[row_mask],
                forced_route_gains=gains[row_mask],
            )
            per_example_loss = F.cross_entropy(
                winner_logits, targets[row_mask], reduction="none"
            )
            row_advantage = best_advantage[row_mask].detach().clamp(
                min=self.min_advantage, max=self.advantage_clip
            )
            importance = row_advantage / row_advantage.mean().clamp_min(1e-8)
            aux_loss = self.weight * (per_example_loss * importance).mean()
            gradients = torch.autograd.grad(
                aux_loss, parameters, allow_unused=False, retain_graph=False
            )
            row_grads = [
                gradient[circuit_id].detach().clone() for gradient in gradients
            ]
            mean_advantage = float(row_advantage.mean().cpu())
            row_updates.append(
                CircuitRowCredit(
                    circuit_id=int(circuit_id),
                    examples=support,
                    mean_advantage=mean_advantage,
                    auxiliary_loss=float(aux_loss.detach().cpu()),
                    down_grad=row_grads[0],
                    up_grad=row_grads[1],
                    bias_grad=row_grads[2],
                )
            )
            self.gradient_forward_examples += support
            total_examples += support
            total_advantage += mean_advantage * support

            if task_ids is not None:
                assigned_tasks = task_ids[row_mask].detach().to("cpu").long()
                valid = assigned_tasks.ge(0) & assigned_tasks.lt(15)
                if valid.any():
                    self.responsibility_counts[:, circuit_id] += torch.bincount(
                        assigned_tasks[valid], minlength=15
                    )

        if not row_updates:
            return None
        return CreditUpdate(
            target_step=int(target_step),
            target_slot=int(target_slot),
            rows=row_updates,
            responsible_examples=total_examples,
            mean_advantage=total_advantage / max(total_examples, 1),
        )

    def apply_update(self, circuits: nn.Module, update: CreditUpdate | None) -> None:
        if update is None:
            return
        for row in update.rows:
            for name, row_grad in (
                ("down", row.down_grad),
                ("up", row.up_grad),
                ("bias", row.bias_grad),
            ):
                parameter = getattr(circuits, name)
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                parameter.grad[row.circuit_id].add_(
                    row_grad.to(parameter.grad.device, parameter.grad.dtype)
                )

            aux_norm = torch.stack([
                row.down_grad.float().pow(2).sum(),
                row.up_grad.float().pow(2).sum(),
                row.bias_grad.float().pow(2).sum(),
            ]).sum().sqrt()
            self.shadow_update_events[row.circuit_id] += 1
            self.shadow_update_examples[row.circuit_id] += row.examples
            self.shadow_advantage_sum[row.circuit_id] += row.mean_advantage * row.examples
            self.shadow_aux_grad_norm_sum[row.circuit_id] += float(aux_norm.cpu())
        self.events_applied += 1
        self.responsible_examples += update.responsible_examples

    def report(self) -> dict[str, Any]:
        usage = self.on_policy_usage
        grad_samples = self.main_grad_sample_count
        grad_nonzero = self.main_grad_nonzero_samples
        shadow_events = self.shadow_update_events
        used = usage.gt(0)
        trained = usage.gt(0) | shadow_events.gt(0)
        gradient_rate = grad_nonzero.float() / grad_samples.clamp_min(1).float()
        undertrained = grad_samples.gt(0) & gradient_rate.lt(0.25) & shadow_events.eq(0)
        niche_nmi, niche_specialization, niche_purity = _nmi_from_counts(
            self.responsibility_counts
        )

        circuits = []
        for circuit_id in range(self.num_circuits):
            events = int(shadow_events[circuit_id])
            examples = int(self.shadow_update_examples[circuit_id])
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
                "shadow_update_examples": examples,
                "shadow_mean_advantage": (
                    float(self.shadow_advantage_sum[circuit_id]) / examples
                    if examples else None
                ),
                "shadow_aux_grad_norm_sum": float(self.shadow_aux_grad_norm_sum[circuit_id]),
                "responsibility_task_counts": self.responsibility_counts[:, circuit_id].tolist(),
            })

        return {
            "events_attempted": self.events_attempted,
            "events_applied": self.events_applied,
            "dead_on_policy_circuits": int((~used).sum()),
            "never_trained_circuits": int((~trained).sum()),
            "undertrained_circuits": int(undertrained.sum()),
            "responsible_examples": self.responsible_examples,
            "responsibility_task_circuit_nmi": niche_nmi,
            "responsibility_weighted_specialization": niche_specialization,
            "responsibility_weighted_task_purity": niche_purity,
            "planned_probe_forward_equivalents_per_step": self.planned_probe_forward_overhead,
            "planned_aux_backward_upper_bound_per_step": self.planned_aux_backward_upper_bound,
            "actual_probe_forward_examples": self.probe_forward_examples,
            "actual_gradient_forward_examples": self.gradient_forward_examples,
            "circuits": circuits,
        }
