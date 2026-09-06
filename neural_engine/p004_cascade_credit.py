from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CascadeProbe:
    step: int
    alternative_ids: torch.Tensor
    probed_mask: torch.Tensor
    cascade_plan: torch.Tensor
    fixed_suffix_plan: torch.Tensor


class CascadeCredit:
    """Training-only on-policy credit for an unchanged hard router.

    A probe changes one selected pair at one recurrent step.  The cascade plan
    leaves every later step at -1, so the existing model reroutes the suffix
    from the counterfactual recurrent state.  Credit is therefore based on the
    final corrected output after the changed state distribution, not on a
    local step score or a frozen suffix.

    The auxiliary ranking loss is intentionally limited to the existing
    HierarchicalRouter key rows.  Circuit parameters, router architecture,
    controller and output head continue to learn only from the normal task
    loss.  This keeps P-004 separate from circuit-bank specialization (P-002)
    and candidate-retrieval architecture work (P-001).
    """

    def __init__(
        self,
        *,
        internal_steps: int,
        active_circuits: int,
        candidate_pool: int,
        interval: int = 4,
        diagnostic_interval: int = 16,
        margin: float = 0.01,
        temperature: float = 0.10,
        weight: float = 0.20,
        seed: int = 0,
    ) -> None:
        if internal_steps < 1 or active_circuits < 1:
            raise ValueError("internal_steps and active_circuits must be positive")
        if candidate_pool <= active_circuits:
            raise ValueError("candidate_pool must exceed active_circuits")
        if interval < 1 or diagnostic_interval < interval:
            raise ValueError("invalid probe intervals")
        if margin < 0 or temperature <= 0 or weight <= 0:
            raise ValueError("invalid cascade-credit hyperparameters")
        self.internal_steps = int(internal_steps)
        self.active_circuits = int(active_circuits)
        self.candidate_pool = int(candidate_pool)
        self.interval = int(interval)
        self.diagnostic_interval = int(diagnostic_interval)
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.weight = float(weight)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

        self.events = 0
        self.probed_examples = 0
        self.preferred_alternative_examples = 0
        self.preferred_current_examples = 0
        self.mean_abs_advantage_sum = 0.0
        self.cascade_advantage_sum = 0.0
        self.fixed_suffix_advantage_sum = 0.0
        self.cascade_shift_sum = 0.0
        self.diagnostic_examples = 0
        self.suffix_overlap_sum = 0.0
        self.suffix_exact_sum = 0.0
        self.suffix_route_observations = 0

    @property
    def planned_extra_forward_equivalents_per_step(self) -> float:
        # One cascade replay each event, plus a fixed-suffix diagnostic replay.
        return 1.0 / self.interval + 1.0 / self.diagnostic_interval

    def should_probe(self, step_number: int) -> bool:
        return step_number > 0 and step_number % self.interval == 0

    def should_diagnose(self, step_number: int) -> bool:
        return step_number > 0 and step_number % self.diagnostic_interval == 0

    def target_step(self, step_number: int) -> int:
        event_index = max(0, step_number // self.interval - 1)
        return event_index % self.internal_steps

    def _sample_alternative(
        self,
        selected: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace one active row with another row from the *same* candidate pool.

        This deliberately avoids P-001 candidate-retrieval work.  The probe asks
        only whether final cascade credit can improve choice among already
        retrieved circuits.
        """
        if selected.ndim != 2 or selected.shape[1] != self.active_circuits:
            raise ValueError("selected must be [batch, active_circuits]")
        if candidates.ndim != 2 or candidates.shape[1] != self.candidate_pool:
            raise ValueError("candidates must be [batch, candidate_pool]")
        batch = selected.shape[0]
        alternative = selected.clone()
        probed = torch.zeros(batch, dtype=torch.bool, device=selected.device)
        for row in range(batch):
            current = set(int(x) for x in selected[row].tolist() if int(x) >= 0)
            options = [int(x) for x in candidates[row].tolist() if int(x) >= 0 and int(x) not in current]
            if not options:
                continue
            option_index = int(torch.randint(len(options), (), generator=self.generator).item())
            side = int(torch.randint(self.active_circuits, (), generator=self.generator).item())
            alternative[row, side] = options[option_index]
            probed[row] = True
        return alternative, probed

    def build_probe(self, stats: dict[str, torch.Tensor], step_number: int) -> CascadeProbe | None:
        if not self.should_probe(step_number):
            return None
        selected_all = stats["selected_ids"].detach()
        candidate_all = stats["candidate_ids"].detach()
        if selected_all.ndim != 3 or candidate_all.ndim != 3:
            raise ValueError("P-004 requires recurrent hard-route statistics")
        step = self.target_step(step_number)
        selected = selected_all[:, step]
        candidates = candidate_all[:, step]
        alternative, probed = self._sample_alternative(selected, candidates)
        if not bool(probed.any()):
            return None

        cascade_plan = torch.full_like(selected_all, -1)
        cascade_plan[probed, step] = alternative[probed]
        fixed_suffix_plan = selected_all.clone()
        fixed_suffix_plan[probed, step] = alternative[probed]
        self.events += 1
        self.probed_examples += int(probed.sum())
        return CascadeProbe(
            step=step,
            alternative_ids=alternative,
            probed_mask=probed,
            cascade_plan=cascade_plan,
            fixed_suffix_plan=fixed_suffix_plan,
        )

    @staticmethod
    def _pair_scores(
        router: nn.Module,
        query: torch.Tensor,
        pair_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(router, "keys"):
            raise TypeError("P-004 cascade credit requires the existing key-based router")
        keys = router.keys[pair_ids]
        scores = torch.einsum("bd,bkd->bk", query, keys) / math.sqrt(query.shape[-1])
        return scores.mean(dim=-1)

    def auxiliary_loss(
        self,
        router: nn.Module,
        query: torch.Tensor,
        current_ids: torch.Tensor,
        alternative_ids: torch.Tensor,
        current_loss: torch.Tensor,
        alternative_loss: torch.Tensor,
        probed_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Advantage-weighted ranking from *final* cascade loss.

        query is detached by the caller/inside this function, so auxiliary
        credit updates existing router key rows only.  A positive advantage
        means the counterfactual route should outrank the current pair; a
        negative advantage reinforces the current pair.
        """
        advantage = (current_loss - alternative_loss).detach()
        valid = probed_mask & advantage.abs().ge(self.margin) & advantage.isfinite()
        if not bool(valid.any()):
            zero = router.keys.sum() * 0.0
            return zero, {
                "credited_examples": 0.0,
                "mean_advantage": 0.0,
                "mean_abs_advantage": 0.0,
            }

        detached_query = query.detach()
        current_score = self._pair_scores(router, detached_query, current_ids)
        alternative_score = self._pair_scores(router, detached_query, alternative_ids)
        preference = advantage.sign()
        score_delta = alternative_score - current_score
        scale = (advantage.abs() / max(self.margin, 1e-6)).clamp(max=4.0)
        ranking = F.softplus(-preference * score_delta / self.temperature) * scale
        loss = self.weight * ranking[valid].mean()

        positive = valid & advantage.gt(0)
        negative = valid & advantage.lt(0)
        self.preferred_alternative_examples += int(positive.sum())
        self.preferred_current_examples += int(negative.sum())
        self.mean_abs_advantage_sum += float(advantage[valid].abs().sum().cpu())
        self.cascade_advantage_sum += float(advantage[valid].sum().cpu())
        return loss, {
            "credited_examples": float(valid.sum()),
            "mean_advantage": float(advantage[valid].mean().cpu()),
            "mean_abs_advantage": float(advantage[valid].abs().mean().cpu()),
        }

    def observe_diagnostic(
        self,
        probe: CascadeProbe,
        natural_loss: torch.Tensor,
        cascade_loss: torch.Tensor,
        fixed_suffix_loss: torch.Tensor,
        natural_stats: dict[str, torch.Tensor],
        cascade_stats: dict[str, torch.Tensor],
    ) -> None:
        mask = probe.probed_mask
        if not bool(mask.any()):
            return
        cascade_adv = (natural_loss - cascade_loss)[mask]
        fixed_adv = (natural_loss - fixed_suffix_loss)[mask]
        self.fixed_suffix_advantage_sum += float(fixed_adv.sum().cpu())
        self.cascade_shift_sum += float((cascade_loss - fixed_suffix_loss)[mask].sum().cpu())
        self.diagnostic_examples += int(mask.sum())

        if probe.step >= self.internal_steps - 1:
            return
        natural_suffix = natural_stats["selected_ids"][mask, probe.step + 1 :].detach()
        cascade_suffix = cascade_stats["selected_ids"][mask, probe.step + 1 :].detach()
        valid = natural_suffix.ge(0) & cascade_suffix.ge(0)
        if not bool(valid.any()):
            return
        overlap = natural_suffix.eq(cascade_suffix).float()
        # selected IDs are top-k ordered; exact-slot overlap is intentionally a
        # strict stability measure.  Exact route requires the whole suffix.
        self.suffix_overlap_sum += float(overlap[valid].sum().cpu())
        self.suffix_route_observations += int(valid.sum())
        per_example_exact = natural_suffix.eq(cascade_suffix).all(dim=-1).all(dim=-1)
        self.suffix_exact_sum += float(per_example_exact.float().sum().cpu())

    def report(self) -> dict[str, Any]:
        credited = self.preferred_alternative_examples + self.preferred_current_examples
        return {
            "events": self.events,
            "probed_examples": self.probed_examples,
            "credited_examples": credited,
            "preferred_alternative_examples": self.preferred_alternative_examples,
            "preferred_current_examples": self.preferred_current_examples,
            "mean_abs_final_ce_advantage": (
                self.mean_abs_advantage_sum / credited if credited else 0.0
            ),
            "mean_signed_final_ce_advantage": (
                self.cascade_advantage_sum / credited if credited else 0.0
            ),
            "mean_fixed_suffix_advantage": (
                self.fixed_suffix_advantage_sum / self.diagnostic_examples
                if self.diagnostic_examples else 0.0
            ),
            "mean_cascade_shift_ce": (
                self.cascade_shift_sum / self.diagnostic_examples
                if self.diagnostic_examples else 0.0
            ),
            "suffix_slot_stability": (
                self.suffix_overlap_sum / self.suffix_route_observations
                if self.suffix_route_observations else 1.0
            ),
            "planned_extra_forward_equivalents_per_step": (
                self.planned_extra_forward_equivalents_per_step
            ),
        }
