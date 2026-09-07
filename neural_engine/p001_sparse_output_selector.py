from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SparseStepContext:
    step: int
    query: torch.Tensor
    state_before: torch.Tensor
    encoded: torch.Tensor
    candidate_ids: torch.Tensor
    route_gain: torch.Tensor


class SparseOutputSignatureSelector(nn.Module):
    """Candidate-only surrogate distilled from the local output-aware oracle.

    Inference never executes the full circuit bank. For each of the M retrieved
    candidates, a tiny rank-r surrogate produces an s-dimensional output
    signature. A shared symmetric pair head scores C(M, 2) pairs. Only the
    winning two real circuits are executed by the frozen Neural Engine body.

    The selector is intentionally separate from retrieval: candidate IDs are
    produced by the existing HierarchicalRouter unchanged.
    """

    def __init__(
        self,
        state_dim: int,
        num_circuits: int,
        candidate_pool: int,
        *,
        active_circuits: int = 2,
        signature_rank: int = 2,
        signature_dim: int = 8,
        key_prior_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if active_circuits != 2:
            raise ValueError("Handoff D selector currently requires active_circuits=2")
        if candidate_pool < 2:
            raise ValueError("candidate_pool must be at least 2")
        if signature_rank < 1 or signature_dim < 2:
            raise ValueError("signature_rank and signature_dim must be positive")
        self.state_dim = int(state_dim)
        self.num_circuits = int(num_circuits)
        self.candidate_pool = int(candidate_pool)
        self.active_circuits = int(active_circuits)
        self.signature_rank = int(signature_rank)
        self.signature_dim = int(signature_dim)
        self.key_prior_weight = float(key_prior_weight)
        if self.key_prior_weight < 0.0:
            raise ValueError("key_prior_weight must be non-negative")

        self.signature_down = nn.Parameter(
            torch.empty(num_circuits, state_dim, signature_rank)
        )
        self.signature_up = nn.Parameter(
            torch.empty(num_circuits, signature_rank, signature_dim)
        )
        self.signature_bias = nn.Parameter(
            torch.zeros(num_circuits, signature_dim)
        )
        self.circuit_bias = nn.Parameter(torch.zeros(num_circuits))
        self.query_projection = nn.Linear(state_dim, signature_dim, bias=False)
        self.pair_head = nn.Sequential(
            nn.Linear(4 * signature_dim, signature_dim),
            nn.GELU(),
            nn.Linear(signature_dim, 1),
        )
        nn.init.normal_(self.signature_down, std=0.02)
        nn.init.normal_(self.signature_up, std=0.02)
        nn.init.normal_(self.query_projection.weight, std=0.02)

        positions = torch.tensor(
            list(itertools.combinations(range(candidate_pool), 2)),
            dtype=torch.long,
        )
        self.register_buffer("pair_positions", positions, persistent=False)
        self.register_buffer("reference_keys", torch.empty(0), persistent=False)

    @torch.no_grad()
    def set_reference_keys(self, keys: torch.Tensor) -> None:
        if keys.shape != (self.num_circuits, self.state_dim):
            raise ValueError(
                "reference keys must have shape "
                f"({self.num_circuits}, {self.state_dim})"
            )
        self.reference_keys = keys.detach().clone()

    @torch.no_grad()
    def initialize_from_circuit_bank(
        self,
        down: torch.Tensor,
        up: torch.Tensor,
        bias: torch.Tensor,
        projection: torch.Tensor,
    ) -> None:
        """Seed the surrogate with a projected copy of the frozen circuit bank.

        This keeps inference candidate-only: the selector still evaluates its
        own compact rows, not the model's full circuit bank.  When the source
        rank matches ``signature_rank``, the initialized signature is the
        exact source circuit output after a fixed low-dimensional projection.
        Later training is allowed to adapt the copy to the local pair target.
        """
        if down.shape[0] != self.num_circuits:
            raise ValueError("down circuit count does not match selector")
        if up.shape[0] != self.num_circuits or bias.shape[0] != self.num_circuits:
            raise ValueError("circuit bank count does not match selector")
        if down.shape[-1] < self.signature_rank or up.shape[-2] < self.signature_rank:
            raise ValueError("source circuit rank is smaller than signature rank")
        if down.shape[1] != self.state_dim or up.shape[-1] != self.state_dim:
            raise ValueError("circuit state dimension does not match selector")
        if bias.shape[-1] != self.state_dim:
            raise ValueError("circuit bias dimension does not match selector")
        if projection.shape != (self.state_dim, self.signature_dim):
            raise ValueError(
                "projection must have shape "
                f"({self.state_dim}, {self.signature_dim})"
            )
        source_down = down[..., : self.signature_rank]
        source_up = up[..., : self.signature_rank, :]
        projected_up = torch.einsum("erk,ks->ers", source_up, projection)
        projected_bias = torch.einsum("ek,ks->es", bias, projection)
        self.signature_down.copy_(source_down)
        self.signature_up.copy_(projected_up)
        self.signature_bias.copy_(projected_bias)

    @property
    def pair_count(self) -> int:
        return int(self.pair_positions.shape[0])

    def candidate_signatures(
        self,
        query: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_ids.shape != (query.shape[0], self.candidate_pool):
            raise ValueError(
                f"candidate_ids must have shape {(query.shape[0], self.candidate_pool)}"
            )
        down = self.signature_down[candidate_ids]
        up = self.signature_up[candidate_ids]
        bias = self.signature_bias[candidate_ids]
        hidden = torch.einsum("bd,bmdr->bmr", query, down)
        hidden = F.gelu(hidden)
        return torch.einsum("bmr,bmrs->bms", hidden, up) + bias

    def forward(
        self,
        query: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signatures = self.candidate_signatures(query, candidate_ids)
        left_pos = self.pair_positions[:, 0]
        right_pos = self.pair_positions[:, 1]
        left = signatures[:, left_pos]
        right = signatures[:, right_pos]
        mean = 0.5 * (left + right)
        product = left * right
        distance = (left - right).abs()
        context = torch.tanh(self.query_projection(query)).unsqueeze(1)
        context = context.expand(-1, self.pair_count, -1)
        features = torch.cat([mean, product, distance, context], dim=-1)
        scores = self.pair_head(features).squeeze(-1)

        left_ids = candidate_ids[:, left_pos]
        right_ids = candidate_ids[:, right_pos]
        scores = scores + 0.5 * (
            self.circuit_bias[left_ids] + self.circuit_bias[right_ids]
        )
        if self.key_prior_weight:
            if self.reference_keys.numel() == 0:
                raise RuntimeError("key prior requested but reference keys are unset")
            key_logits = torch.einsum(
                "bd,bmd->bm", query, self.reference_keys[candidate_ids]
            ) / math.sqrt(self.state_dim)
            key_pair = 0.5 * (
                key_logits[:, left_pos] + key_logits[:, right_pos]
            )
            scores = scores + self.key_prior_weight * key_pair
        pair_ids = torch.stack([left_ids, right_ids], dim=-1)
        return scores, pair_ids

    @torch.no_grad()
    def select(
        self,
        query: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores, pair_ids = self(query, candidate_ids)
        index = scores.argmax(dim=-1)
        rows = torch.arange(query.shape[0], device=query.device)
        return pair_ids[rows, index], scores, index

    def touched_parameter_report(self) -> dict[str, int | float]:
        r = self.signature_rank
        s = self.signature_dim
        d = self.state_dim
        m = self.candidate_pool
        per_candidate = d * r + r * s + s + 1
        candidate_rows = m * per_candidate
        shared = self.query_projection.weight.numel()
        shared += sum(parameter.numel() for parameter in self.pair_head.parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        return {
            "selector_total_params": int(total),
            "selector_candidate_rows_touched_per_decision": m,
            "selector_candidate_touched_params_per_decision": int(candidate_rows),
            "selector_shared_touched_params_per_decision": int(shared),
            "selector_total_touched_params_per_decision": int(candidate_rows + shared),
            "selector_pair_scores_per_decision": self.pair_count,
            "dense_bank_circuit_outputs_per_decision": 0,
        }


class ExistingKeyPairSelector(nn.Module):
    """Reference selector used to verify custom rollout parity."""

    def __init__(self, keys: torch.Tensor, candidate_pool: int) -> None:
        super().__init__()
        self.keys = keys
        self.candidate_pool = int(candidate_pool)

    def select(
        self,
        query: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidate_keys = self.keys[candidate_ids]
        logits = torch.einsum("bd,bmd->bm", query, candidate_keys)
        logits = logits / math.sqrt(query.shape[-1])
        values, positions = logits.topk(2, dim=-1)
        selected = candidate_ids.gather(1, positions)
        return selected, values, positions[:, 0]


def validate_plain_sparse_contract(model: nn.Module) -> None:
    if getattr(model, "router_variant", None) != "global":
        raise ValueError("sparse output-signature experiment requires global router")
    if model.router.__class__.__name__ != "HierarchicalRouter":
        raise ValueError("existing HierarchicalRouter must remain unchanged")
    if model.active_circuits != 2:
        raise ValueError("experiment requires active_circuits=2")
    if getattr(model, "circuit_mode", "parallel") != "parallel":
        raise ValueError("experiment requires parallel circuit execution")
    if getattr(model, "use_task_context", False):
        raise ValueError("task-context variants are outside Handoff D scope")
    if getattr(model, "memory_write", None) is not None:
        raise ValueError("memory-write variants are outside Handoff D scope")
    if getattr(model, "adaptive_halting", False):
        raise ValueError("adaptive halting is outside Handoff D scope")
    if getattr(model.router, "soft_routing_temperature", 0.0) != 0.0:
        raise ValueError("experiment requires hard sparse routing")


def selected_pair_weights(
    model: nn.Module,
    query: torch.Tensor,
    selected_ids: torch.Tensor,
) -> torch.Tensor:
    keys = model.router.keys[selected_ids]
    logits = torch.einsum("bd,bkd->bk", query, keys) / math.sqrt(query.shape[-1])
    return F.softmax(logits, dim=-1)


@torch.no_grad()
def sparse_selector_rollout(
    model: nn.Module,
    selector: SparseOutputSignatureSelector | ExistingKeyPairSelector,
    inputs: torch.Tensor,
    *,
    return_contexts: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[SparseStepContext]]:
    """Run the unchanged model body with a candidate-only external selector."""
    validate_plain_sparse_contract(model)
    encoded = model.encode(inputs)
    state = model.state.initialize(encoded)
    batch = inputs.shape[0]
    selected_steps: list[torch.Tensor] = []
    candidate_steps: list[torch.Tensor] = []
    query_steps: list[torch.Tensor] = []
    weight_steps: list[torch.Tensor] = []
    gain_steps: list[torch.Tensor] = []
    contexts: list[SparseStepContext] = []
    last_logits = model.output(state)

    for step in range(model.internal_steps):
        query = state + model.step_embedding[step]
        _, _, route_stats = model.router(
            query,
            coverage=False,
            exploration_prob=0.0,
        )
        candidates = route_stats["candidate_ids"]
        selected, _, _ = selector.select(query, candidates)
        weights = selected_pair_weights(model, query, selected)
        route_gain = route_stats["route_gain"]
        if return_contexts:
            contexts.append(
                SparseStepContext(
                    step=step,
                    query=query.detach(),
                    state_before=state.detach(),
                    encoded=encoded.detach(),
                    candidate_ids=candidates.detach(),
                    route_gain=route_gain.detach(),
                )
            )
        circuit_delta = model.circuits(query, selected, weights)
        update = (
            circuit_delta * route_gain.unsqueeze(-1)
            + model.input_reinjection * encoded
            + model.step_embedding[step]
        )
        state = model.state.step(state, update)
        last_logits = model.output(state)
        selected_steps.append(selected)
        candidate_steps.append(candidates)
        query_steps.append(query)
        weight_steps.append(weights)
        gain_steps.append(route_gain)

    stats = {
        "selected_ids": torch.stack(selected_steps, dim=1),
        "candidate_ids": torch.stack(candidate_steps, dim=1),
        "query_states": torch.stack(query_steps, dim=1),
        "selected_weights": torch.stack(weight_steps, dim=1),
        "route_gains": torch.stack(gain_steps, dim=1),
    }
    return last_logits, stats, contexts


def _candidate_real_outputs(
    model: nn.Module,
    query: torch.Tensor,
    candidate_ids: torch.Tensor,
) -> torch.Tensor:
    down = model.circuits.down[candidate_ids]
    up = model.circuits.up[candidate_ids]
    bias = model.circuits.bias[candidate_ids]
    hidden = torch.einsum("bd,bmdr->bmr", query, down)
    hidden = F.gelu(hidden)
    return torch.einsum("bmr,bmrd->bmd", hidden, up) + bias


@torch.no_grad()
def candidate_local_teacher(
    model: nn.Module,
    context: SparseStepContext,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Candidate-only dense teacher used during training/diagnostics only.

    It executes the real circuit body for M candidates, never the full E bank.
    Pair losses use the real immediate GRU transition and output head. The
    returned loss matrix has one column for every C(M,2) candidate pair.
    """
    validate_plain_sparse_contract(model)
    query = context.query
    candidates = context.candidate_ids
    outputs = _candidate_real_outputs(model, query, candidates)
    pair_positions = torch.tensor(
        list(itertools.combinations(range(candidates.shape[1]), 2)),
        device=query.device,
        dtype=torch.long,
    )
    left_pos = pair_positions[:, 0]
    right_pos = pair_positions[:, 1]
    key_logits = torch.einsum(
        "bd,bmd->bm", query, model.router.keys[candidates]
    ) / math.sqrt(query.shape[-1])
    pair_key_logits = torch.stack(
        [key_logits[:, left_pos], key_logits[:, right_pos]], dim=-1
    )
    pair_weights = F.softmax(pair_key_logits, dim=-1)
    pair_outputs = torch.stack(
        [outputs[:, left_pos], outputs[:, right_pos]], dim=-2
    )
    pair_delta = (pair_outputs * pair_weights.unsqueeze(-1)).sum(dim=-2)
    update = (
        pair_delta * context.route_gain.view(-1, 1, 1)
        + model.input_reinjection * context.encoded.unsqueeze(1)
        + model.step_embedding[context.step].view(1, 1, -1)
    )
    batch, pairs, dim = update.shape
    flat_state = context.state_before.unsqueeze(1).expand(-1, pairs, -1).reshape(-1, dim)
    proposal = model.state.step(flat_state, update.reshape(-1, dim))
    local_logits = model.output(proposal).reshape(batch, pairs, -1)
    repeated_targets = targets.view(-1, 1).expand(-1, pairs).reshape(-1)
    full_losses = F.cross_entropy(
        local_logits.reshape(-1, local_logits.shape[-1]),
        repeated_targets,
        reduction="none",
    ).reshape(batch, pairs)

    # Individual-output additive ablation: each candidate is evaluated alone
    # through the same GRU+head, then pair cost is the mean of its members.
    single_update = (
        outputs * context.route_gain.view(-1, 1, 1)
        + model.input_reinjection * context.encoded.unsqueeze(1)
        + model.step_embedding[context.step].view(1, 1, -1)
    )
    m = candidates.shape[1]
    single_state = context.state_before.unsqueeze(1).expand(-1, m, -1).reshape(-1, dim)
    single_proposal = model.state.step(single_state, single_update.reshape(-1, dim))
    single_logits = model.output(single_proposal).reshape(batch, m, -1)
    single_targets = targets.view(-1, 1).expand(-1, m).reshape(-1)
    single_losses = F.cross_entropy(
        single_logits.reshape(-1, single_logits.shape[-1]),
        single_targets,
        reduction="none",
    ).reshape(batch, m)
    additive_losses = 0.5 * (
        single_losses[:, left_pos] + single_losses[:, right_pos]
    )

    # No-GRU ablation: keep the same pair output and class head, but replace the
    # learned recurrent transition with a simple residual state update.
    residual_state = context.state_before.unsqueeze(1) + update
    residual_logits = model.output(residual_state.reshape(-1, dim)).reshape(batch, pairs, -1)
    residual_losses = F.cross_entropy(
        residual_logits.reshape(-1, residual_logits.shape[-1]),
        repeated_targets,
        reduction="none",
    ).reshape(batch, pairs)

    # Head-free diagnostic: magnitude of the real GRU state change. This is not
    # a deployable task score; it only measures how much of the proxy advantage
    # disappears when the output head/label alignment is removed.
    proposal_3d = proposal.reshape(batch, pairs, dim)
    state_change = (proposal_3d - context.state_before.unsqueeze(1)).pow(2).mean(dim=-1)

    return {
        "full_local_losses": full_losses,
        "individual_additive_losses": additive_losses,
        "no_gru_losses": residual_losses,
        "gru_state_change": state_change,
        "pair_positions": pair_positions,
    }


def distillation_loss(
    student_scores: torch.Tensor,
    teacher_losses: torch.Tensor,
    *,
    temperature: float = 0.10,
    hard_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= hard_weight <= 1.0:
        raise ValueError("hard_weight must be in [0, 1]")
    teacher_prob = F.softmax(-teacher_losses.detach() / temperature, dim=-1)
    student_log_prob = F.log_softmax(student_scores / temperature, dim=-1)
    soft = -(teacher_prob * student_log_prob).sum(dim=-1).mean()
    target = teacher_losses.detach().argmin(dim=-1)
    hard = F.cross_entropy(student_scores, target)
    loss = (1.0 - hard_weight) * soft + hard_weight * hard
    with torch.no_grad():
        student_index = student_scores.argmax(dim=-1)
        teacher_index = target
        rows = torch.arange(student_scores.shape[0], device=student_scores.device)
        regret = (
            teacher_losses[rows, student_index] - teacher_losses[rows, teacher_index]
        ).clamp_min(0)
    return loss, {
        "teacher_pair_match": float(student_index.eq(teacher_index).float().mean().cpu()),
        "teacher_local_regret": float(regret.mean().cpu()),
    }


def selector_cost_report(
    model: nn.Module,
    selector: SparseOutputSignatureSelector,
) -> dict[str, Any]:
    report = selector.touched_parameter_report()
    d = int(model.state_dim)
    m = int(model.router.candidate_pool)
    r = int(selector.signature_rank)
    s = int(selector.signature_dim)
    pairs = int(selector.pair_count)
    report.update({
        "signature_muladds_per_decision": int(m * (d * r + r * s)),
        "query_projection_muladds_per_decision": int(d * s),
        "pair_head_approx_muladds_per_decision": int(pairs * (4 * s * s + s)),
        "real_circuit_rows_executed_per_decision": int(model.active_circuits),
        "full_bank_real_circuit_rows_scored_per_decision": 0,
        "candidate_real_circuit_rows_scored_at_inference": 0,
    })
    key_rows = int(model.router.candidate_pool * model.state_dim) if selector.key_prior_weight else 0
    report["key_prior_weight"] = float(selector.key_prior_weight)
    report["key_prior_touched_params_per_decision"] = key_rows
    report["selector_total_touched_params_per_decision"] += key_rows
    return report


__all__ = [
    "SparseOutputSignatureSelector",
    "ExistingKeyPairSelector",
    "SparseStepContext",
    "sparse_selector_rollout",
    "candidate_local_teacher",
    "distillation_loss",
    "selector_cost_report",
    "validate_plain_sparse_contract",
]
