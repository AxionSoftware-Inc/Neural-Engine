"""Opt-in compact cost router; no changes to NeuralEngineV0 or its body.

Replace ``model.router`` with CoupledProbeRouter(...). The parent supplies the
existing routing/statistics API. Unlike ProbeRouteRouter, retrieval and pair
selection share a utility: signed probe feedback can reach an excluded key.
The interaction is a predictor, not an exact decomposition of circuit costs.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .router import ProbeRouteRouter


class CoupledProbeRouter(ProbeRouteRouter):
    def __init__(self, state_dim: int, num_circuits: int, branch: int = 8,
                 depth: int = 4, candidate_pool: int = 8,
                 active_circuits: int = 2, num_addresses: int = 1,
                 routing_capacity: int | None = None,
                 routing_depth: int | None = None, pair_rank: int = 8,
                 router_dim: int = 32):
        if router_dim < 1:
            raise ValueError("router_dim must be positive")
        super().__init__(state_dim, num_circuits, branch, depth, candidate_pool,
                         active_circuits, num_addresses, routing_capacity,
                         routing_depth, pair_rank)
        # Do not retain the parent's two independent D x D projections/keys.
        del self.utility_query, self.utility_keys
        self.retriever_query = nn.Linear(state_dim, router_dim, bias=False)
        self.keys = nn.Parameter(torch.empty(num_circuits, router_dim))
        self.interaction_query = nn.Linear(router_dim, pair_rank, bias=False)
        nn.init.normal_(self.keys, std=0.1)
        nn.init.normal_(self.pair_embeddings, std=1.0 / math.sqrt(pair_rank))
        nn.init.zeros_(self.interaction_query.weight)

    def _features(self, state: torch.Tensor) -> torch.Tensor:
        # Non-affine normalization, no attention and no task-ID input.
        return F.gelu(self.retriever_query(F.layer_norm(state, (state.shape[-1],))))

    def _capacity(self, capacity: int | None) -> int:
        capacity = self.routing_capacity if capacity is None else int(capacity)
        if not self.candidate_pool <= capacity <= self.routing_capacity:
            raise ValueError("capacity must cover candidate_pool and not exceed routing capacity")
        return capacity

    def retriever_scores(self, state: torch.Tensor, capacity: int | None = None) -> torch.Tensor:
        capacity = self._capacity(capacity)
        query = self._features(state)
        return query @ self.keys[:capacity].T / math.sqrt(self.keys.shape[-1])

    def utility_scores(self, state: torch.Tensor, capacity: int | None = None) -> torch.Tensor:
        return self.retriever_scores(state, capacity)

    def _pair_scores(self, state: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        features = self._features(state)
        utility = features @ self.keys[:self.routing_capacity].T / math.sqrt(self.keys.shape[-1])
        left, right = pairs[..., 0], pairs[..., 1]
        additive = utility.gather(1, left) + utility.gather(1, right)
        interaction = torch.tanh(self.interaction_query(features)).unsqueeze(1)
        products = self.pair_embeddings[left] * self.pair_embeddings[right]
        return additive + (interaction * products).sum(-1) / math.sqrt(self.pair_rank)

    def selection_scores(self, state: torch.Tensor, pair_ids: torch.Tensor) -> torch.Tensor:
        if pair_ids.shape != (state.shape[0], 2):
            raise ValueError("pair_ids must have shape [batch, 2]")
        if bool(((pair_ids < 0) | (pair_ids >= self.routing_capacity)).any()):
            raise ValueError("pair IDs must be in the active bank; replay sentinels are not scores")
        if bool(pair_ids[:, 0].eq(pair_ids[:, 1]).any()):
            raise ValueError("pairs must contain distinct circuits")
        return self._pair_scores(state, pair_ids.unsqueeze(1)).squeeze(1)

    def candidate_pair_scores(self, state: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        if candidate_ids.shape != (state.shape[0], self.candidate_pool):
            raise ValueError("candidate_ids must have shape [batch, candidate_pool]")
        pairs = candidate_ids[:, self.pair_positions]
        return self._pair_scores(state, pairs)

    def forward(self, state: torch.Tensor, coverage: bool = False,
                coverage_temperature: float = 0.25, exploration_prob: float = 0.0,
                routing_offset: int | torch.Tensor = 0,
                routing_capacity: int | None = None,
                routing_windows: torch.Tensor | None = None,
                target_bases: torch.Tensor | None = None):
        capacity = self._capacity(routing_capacity)
        if not 0 <= exploration_prob <= 1:
            raise ValueError("exploration_prob must be between 0 and 1")
        selected, weights, stats = super().forward(
            state, coverage=False, coverage_temperature=coverage_temperature,
            exploration_prob=0.0, routing_offset=routing_offset,
            routing_capacity=capacity, routing_windows=routing_windows,
            target_bases=target_bases)
        if exploration_prob and self.training:
            # Uniform unordered pairs, without duplicate-repair sampling bias.
            explore = torch.rand(state.shape[0], device=state.device) < exploration_prob
            first = torch.randint(capacity, (state.shape[0],), device=state.device)
            second = torch.randint(capacity - 1, first.shape, device=state.device)
            second = second + second.ge(first)
            sampled = torch.stack((first, second), -1)
            selected = torch.where(explore[:, None], sampled, selected)
            stats["selected_ids"] = selected
            stats["exploration_mask"] = explore
        if coverage:
            probabilities = F.softmax(stats["retriever_logits"] / coverage_temperature, -1)
            distribution = probabilities.mean(0)
            stats["routing_coverage_loss"] = (
                distribution * (distribution.clamp_min(1e-8).log() + math.log(capacity))
            ).sum()
        return selected, weights, stats


def signed_probe_loss(router, query: torch.Tensor, current: torch.Tensor,
                      alternative: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Cost differences in CE units, not token-wise normalized pseudo regret.

The coarse additive target is an explicit approximation. Its quality must be
checked with exact candidate recall/regret, not inferred from this loss.
All labels/queries are detached; no probe branch trains the frozen body.
"""
    query, delta = query.detach(), delta.detach()
    if delta.numel() == 0:
        zero = next(router.parameters()).sum() * 0
        return zero, {"selection": zero.detach(), "retrieval": zero.detach()}
    prediction = (router.selection_scores(query, alternative)
                  - router.selection_scores(query, current))
    utility = router.retriever_scores(query)
    coarse = utility.gather(1, alternative).sum(-1) - utility.gather(1, current).sum(-1)
    selection = F.smooth_l1_loss(prediction, delta, beta=0.1)
    weight = (delta.abs() / 0.1).clamp_max(2) * delta.abs().ge(0.02)
    selection = selection + 0.1 * (
        weight * F.softplus(-delta.sign() * prediction / 0.1)).mean()
    retrieval = F.smooth_l1_loss(coarse, delta, beta=0.1)
    return selection + 0.25 * retrieval, {
        "selection": selection.detach(), "retrieval": retrieval.detach()}
