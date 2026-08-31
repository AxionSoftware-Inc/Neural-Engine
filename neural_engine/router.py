from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class HierarchicalRouter(nn.Module):
    """Route through a small branching tree, then score a local candidate pool."""

    def __init__(self, state_dim: int, num_circuits: int, branch: int = 8, depth: int = 4,
                 candidate_pool: int = 32, active_circuits: int = 8, num_addresses: int = 1):
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
        self.level_projections = nn.Parameter(torch.empty(num_addresses, depth, state_dim, branch))
        self.level_bias = nn.Parameter(torch.zeros(num_addresses, depth, branch))
        self.keys = nn.Parameter(torch.empty(num_circuits, state_dim))
        nn.init.normal_(self.level_projections, std=0.02)
        nn.init.normal_(self.keys, std=0.02)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch = state.shape[0]
        leaf = torch.zeros(batch, self.num_addresses, dtype=torch.long, device=state.device)
        entropies = []
        path_scores = []
        for address in range(self.num_addresses):
            address_leaf = leaf[:, address]
            address_score = torch.zeros(batch, device=state.device)
            for level in range(self.depth):
                logits = state @ self.level_projections[address, level] + self.level_bias[address, level]
                probs = F.softmax(logits, dim=-1)
                entropies.append(-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1))
                child = logits.argmax(dim=-1)
                address_score = address_score + logits.gather(1, child.unsqueeze(1)).squeeze(1)
                address_leaf = address_leaf * self.branch + child
            leaf[:, address] = address_leaf
            path_scores.append(address_score)

        base = leaf.remainder(self.num_circuits)
        offsets = torch.arange(self.candidates_per_address, device=state.device).view(1, 1, -1)
        candidate_ids = (base.unsqueeze(-1) + offsets).remainder(self.num_circuits)
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
            "router_decisions": torch.tensor(self.depth * self.num_addresses, device=state.device),
            "route_gain": route_gain,
            "candidate_ids": candidate_ids,
            "selected_ids": selected_ids,
        }
        return selected_ids, weights, stats
