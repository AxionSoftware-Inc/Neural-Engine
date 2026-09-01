from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class HierarchicalRouter(nn.Module):
    """Route through a small branching tree, then score a local candidate pool."""

    def __init__(self, state_dim: int, num_circuits: int, branch: int = 8, depth: int = 4,
                 candidate_pool: int = 32, active_circuits: int = 8, num_addresses: int = 1,
                 routing_capacity: int | None = None, routing_depth: int | None = None):
        super().__init__()
        if active_circuits > candidate_pool:
            raise ValueError("active_circuits cannot exceed candidate_pool")
        if candidate_pool % num_addresses != 0:
            raise ValueError("candidate_pool must be divisible by num_addresses")
        self.num_circuits = num_circuits
        self.branch = branch
        self.depth = depth
        self.candidate_pool = candidate_pool
        self.active_circuits = active_circuits
        self.num_addresses = num_addresses
        self.candidates_per_address = candidate_pool // num_addresses
        self.routing_capacity = num_circuits if routing_capacity is None else int(routing_capacity)
        self.active_depth = depth if routing_depth is None else int(routing_depth)
        if not 0 < self.routing_capacity <= num_circuits:
            raise ValueError("routing_capacity must be between 1 and num_circuits")
        if self.routing_capacity < candidate_pool:
            raise ValueError("routing_capacity must be at least candidate_pool")
        if not 0 < self.active_depth <= depth:
            raise ValueError("routing_depth must be between 1 and depth")
        self.level_projections = nn.Parameter(torch.empty(num_addresses, depth, state_dim, branch))
        self.level_bias = nn.Parameter(torch.zeros(num_addresses, depth, branch))
        self.keys = nn.Parameter(torch.empty(num_circuits, state_dim))
        nn.init.normal_(self.level_projections, std=0.02)
        nn.init.normal_(self.keys, std=0.02)

    def set_routing_state(self, *, capacity: int | None = None,
                          depth: int | None = None) -> None:
        """Change the reachable bank prefix/tree depth for capacity growth.

        Growth experiments can start with a smaller parent routing geometry,
        then expose the newly initialized rows and tree level after warmup.
        """
        next_capacity = self.routing_capacity if capacity is None else int(capacity)
        next_depth = self.active_depth if depth is None else int(depth)
        if not 0 < next_capacity <= self.num_circuits:
            raise ValueError("routing capacity must be between 1 and num_circuits")
        if next_capacity < self.candidate_pool:
            raise ValueError("routing capacity must be at least candidate_pool")
        if not 0 < next_depth <= self.depth:
            raise ValueError("routing depth must be between 1 and depth")
        self.routing_capacity = next_capacity
        self.active_depth = next_depth

    def _soft_coverage_distribution(self, level_probs: list[torch.Tensor]) -> torch.Tensor:
        """Estimate circuit usage through the soft tree paths.

        Execution still follows the hard argmax path below.  This auxiliary
        distribution is only used during training, so the router can receive
        a differentiable signal before a leaf becomes the hard winner.
        """
        path_probs = level_probs[0]
        for probs in level_probs[1:]:
            path_probs = (path_probs.unsqueeze(-1) * probs.unsqueeze(1)).reshape(path_probs.shape[0], -1)

        # Each leaf exposes the same local candidate window used by the hard
        # route.  Spread its soft mass uniformly across that window to obtain
        # a differentiable approximation of bank-level traffic.
        leaf_mass = path_probs.mean(dim=0)
        leaf_ids = torch.arange(path_probs.shape[-1], device=path_probs.device)
        offsets = torch.arange(self.candidates_per_address, device=path_probs.device).view(1, -1)
        candidate_ids = (leaf_ids.unsqueeze(-1) + offsets).remainder(self.routing_capacity)
        candidate_weights = leaf_mass.unsqueeze(-1).expand(-1, self.candidates_per_address)
        distribution = torch.zeros(self.num_circuits, device=path_probs.device, dtype=path_probs.dtype)
        distribution.scatter_add_(0, candidate_ids.reshape(-1),
                                  (candidate_weights / self.candidates_per_address).reshape(-1))
        return distribution / distribution.sum().clamp_min(1e-8)

    def forward(self, state: torch.Tensor, coverage: bool = False,
                coverage_temperature: float = 0.25,
                exploration_prob: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if coverage_temperature <= 0:
            raise ValueError("coverage_temperature must be positive")
        if not 0.0 <= exploration_prob <= 1.0:
            raise ValueError("exploration_prob must be between 0 and 1")
        batch = state.shape[0]
        leaf = torch.zeros(batch, self.num_addresses, dtype=torch.long, device=state.device)
        entropies = []
        path_scores = []
        coverage_distributions = []
        for address in range(self.num_addresses):
            address_leaf = leaf[:, address]
            address_score = torch.zeros(batch, device=state.device)
            level_probs = []
            coverage_level_probs = []
            for level in range(self.active_depth):
                logits = state @ self.level_projections[address, level] + self.level_bias[address, level]
                probs = F.softmax(logits, dim=-1)
                level_probs.append(probs)
                if coverage:
                    coverage_level_probs.append(F.softmax(logits / coverage_temperature, dim=-1))
                entropies.append(-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1))
                child = logits.argmax(dim=-1)
                if exploration_prob and self.training:
                    explore = torch.rand(batch, device=state.device) < exploration_prob
                    random_child = torch.randint(self.branch, (batch,), device=state.device)
                    child = torch.where(explore, random_child, child)
                address_score = address_score + logits.gather(1, child.unsqueeze(1)).squeeze(1)
                address_leaf = address_leaf * self.branch + child
            leaf[:, address] = address_leaf
            path_scores.append(address_score)
            if coverage:
                coverage_distributions.append(self._soft_coverage_distribution(coverage_level_probs))

        base = leaf.remainder(self.routing_capacity)
        offsets = torch.arange(self.candidates_per_address, device=state.device).view(1, 1, -1)
        candidate_ids = (base.unsqueeze(-1) + offsets).remainder(self.routing_capacity)
        candidate_ids = candidate_ids.reshape(batch, self.candidate_pool)
        candidate_keys = self.keys[candidate_ids]
        candidate_logits = torch.einsum("bd,bkd->bk", state, candidate_keys) / math.sqrt(state.shape[-1])
        top_values, top_positions = candidate_logits.topk(self.active_circuits, dim=-1)
        selected_ids = candidate_ids.gather(1, top_positions)
        # The address is hard/structured for execution, but this small gain
        # keeps the chosen tree path connected to the task loss for learning.
        path_score = torch.stack(path_scores, dim=-1).mean(dim=-1)
        route_gain = 1.0 + 0.05 * torch.tanh(path_score)
        weights = F.softmax(top_values, dim=-1)
        stats = {
            "router_entropy": torch.stack(entropies, dim=-1).mean(),
            "router_decisions": torch.tensor(self.active_depth * self.num_addresses, device=state.device),
            "route_gain": route_gain,
            "candidate_ids": candidate_ids,
            "selected_ids": selected_ids,
        }
        if coverage:
            distribution = torch.stack(coverage_distributions, dim=0).mean(dim=0)
            stats["routing_coverage_loss"] = (
                distribution * (distribution.clamp_min(1e-8).log() + math.log(self.num_circuits))
            ).sum()
        return selected_ids, weights, stats
