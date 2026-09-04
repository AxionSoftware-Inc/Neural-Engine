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
                exploration_prob: float = 0.0,
                routing_offset: int | torch.Tensor = 0,
                routing_capacity: int | None = None,
                routing_windows: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if coverage_temperature <= 0:
            raise ValueError("coverage_temperature must be positive")
        if not 0.0 <= exploration_prob <= 1.0:
            raise ValueError("exploration_prob must be between 0 and 1")
        batch = state.shape[0]
        local_capacity = self.routing_capacity if routing_capacity is None else int(routing_capacity)
        if not 0 < local_capacity <= self.routing_capacity:
            raise ValueError("routing_capacity must be between 1 and the configured routing capacity")
        if routing_windows is not None:
            if not isinstance(routing_offset, int) or routing_offset != 0:
                raise ValueError("routing_windows cannot be combined with a non-zero routing_offset")
            if routing_windows.ndim != 2 or routing_windows.shape[0] != batch:
                raise ValueError("routing_windows must have shape [batch, windows]")
            window_count = routing_windows.shape[1]
            if window_count < 1 or self.candidate_pool % window_count != 0:
                raise ValueError("candidate_pool must be divisible by the number of routing windows")
            window_pool = self.candidate_pool // window_count
            if window_pool % self.num_addresses != 0:
                raise ValueError("each routing window must divide across router addresses")
            routing_windows = routing_windows.to(device=state.device, dtype=torch.long)
            if (routing_windows < 0).any() or (
                    routing_windows + local_capacity > self.num_circuits).any():
                raise ValueError("routing_windows must identify valid bank windows")
        elif isinstance(routing_offset, int):
            if not 0 <= routing_offset <= self.num_circuits - local_capacity:
                raise ValueError("routing_offset must identify a valid bank window")
        else:
            if routing_offset.ndim != 1 or routing_offset.shape[0] != batch:
                raise ValueError("tensor routing_offset must have one value per batch item")
            if (routing_offset < 0).any() or (routing_offset + local_capacity > self.num_circuits).any():
                raise ValueError("tensor routing_offset must identify valid bank windows")
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

        base = leaf.remainder(local_capacity)
        if routing_windows is None:
            offsets = torch.arange(self.candidates_per_address, device=state.device).view(1, 1, -1)
            candidate_ids = (base.unsqueeze(-1) + offsets).remainder(local_capacity)
            if isinstance(routing_offset, int):
                candidate_ids = candidate_ids + routing_offset
            else:
                candidate_ids = candidate_ids + routing_offset.to(state.device).view(batch, 1, 1)
        else:
            window_pool = self.candidate_pool // routing_windows.shape[1]
            window_candidates_per_address = window_pool // self.num_addresses
            offsets = torch.arange(window_candidates_per_address, device=state.device).view(1, 1, 1, -1)
            candidate_ids = (base.unsqueeze(-1).unsqueeze(-1) + offsets).remainder(local_capacity)
            candidate_ids = candidate_ids + routing_windows.view(batch, 1, -1, 1)
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


class StableFamilyRouter(nn.Module):
    """Route within a stable operator/stage family plus a shared fallback.

    The global hierarchical router makes every bank row compete for the same
    candidate window.  That is useful for a compact bank, but it lets a larger
    bank fragment route reuse.  This router fixes the first routing decision
    to a semantic family supplied by the register graph, then performs
    value-dependent routing only inside that family's block.  A second window
    always comes from a small shared block so a local family route can fall
    back to reusable computation.

    The physical layout is stable across model sizes:

        [shared fallback][family 0][family 1]...[family N-1]

    Family blocks are contiguous and balanced.  Growing a bank can therefore
    preserve family identity instead of changing the meaning of every global
    circuit index.
    """

    def __init__(self, state_dim: int, num_circuits: int, branch: int = 8,
                 depth: int = 4, candidate_pool: int = 32,
                 active_circuits: int = 8, num_addresses: int = 1,
                 routing_capacity: int | None = None,
                 routing_depth: int | None = None, family_count: int = 9,
                 shared_fraction: float = 0.125):
        super().__init__()
        if active_circuits > candidate_pool:
            raise ValueError("active_circuits cannot exceed candidate_pool")
        if num_addresses != 1:
            raise ValueError("StableFamilyRouter currently supports one address")
        if candidate_pool % 2:
            raise ValueError("candidate_pool must be even for local and shared windows")
        if family_count < 1:
            raise ValueError("family_count must be positive")
        if not 0.05 <= shared_fraction < 0.5:
            raise ValueError("shared_fraction must be between 0.05 and 0.5")
        if family_count >= num_circuits:
            raise ValueError("family_count must be smaller than num_circuits")

        self.num_circuits = num_circuits
        self.branch = branch
        self.depth = depth
        self.candidate_pool = candidate_pool
        self.active_circuits = active_circuits
        self.num_addresses = num_addresses
        self.family_count = family_count
        self.shared_fraction = shared_fraction
        self.shared_candidate_pool = candidate_pool // 2
        self.local_candidate_pool = candidate_pool - self.shared_candidate_pool
        requested_shared = round(num_circuits * shared_fraction)
        self.shared_count = max(self.shared_candidate_pool,
                                min(num_circuits - family_count,
                                    requested_shared))
        family_total = num_circuits - self.shared_count
        if family_total < family_count * self.local_candidate_pool:
            raise ValueError("each family must fit the local candidate pool")
        base_size, remainder = divmod(family_total, family_count)
        self.family_sizes = tuple(
            base_size + (1 if index < remainder else 0)
            for index in range(family_count))
        offsets = [self.shared_count]
        for size in self.family_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        self.family_offsets = tuple(offsets)

        self.routing_capacity = (num_circuits if routing_capacity is None
                                 else int(routing_capacity))
        self.active_depth = depth if routing_depth is None else int(routing_depth)
        if not 0 < self.routing_capacity <= num_circuits:
            raise ValueError("routing_capacity must be between 1 and num_circuits")
        if self.routing_capacity != num_circuits:
            raise ValueError("StableFamilyRouter requires the full stable bank")
        if not 0 < self.active_depth <= depth:
            raise ValueError("routing_depth must be between 1 and depth")
        max_family_size = max(self.family_sizes)
        local_depth = max(1, math.ceil(math.log(max_family_size, branch)))
        self.local_depth = min(self.active_depth, local_depth)

        self.level_projections = nn.Parameter(
            torch.empty(num_addresses, depth, state_dim, branch))
        self.level_bias = nn.Parameter(torch.zeros(num_addresses, depth, branch))
        self.keys = nn.Parameter(torch.empty(num_circuits, state_dim))
        self.family_embeddings = nn.Parameter(torch.empty(family_count, state_dim))
        nn.init.normal_(self.level_projections, std=0.02)
        nn.init.normal_(self.keys, std=0.02)
        nn.init.normal_(self.family_embeddings, std=0.02)

    def set_routing_state(self, *, capacity: int | None = None,
                          depth: int | None = None) -> None:
        next_capacity = self.routing_capacity if capacity is None else int(capacity)
        next_depth = self.active_depth if depth is None else int(depth)
        if next_capacity != self.num_circuits:
            raise ValueError("StableFamilyRouter requires the full stable bank")
        if not 0 < next_depth <= self.depth:
            raise ValueError("routing_depth must be between 1 and depth")
        self.routing_capacity = next_capacity
        self.active_depth = next_depth
        max_family_size = max(self.family_sizes)
        self.local_depth = min(
            self.active_depth, max(1, math.ceil(math.log(max_family_size, self.branch))))

    def forward(self, state: torch.Tensor, family_ids: torch.Tensor,
                coverage: bool = False, coverage_temperature: float = 0.25,
                exploration_prob: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if coverage:
            raise NotImplementedError("StableFamilyRouter coverage is not implemented")
        if coverage_temperature <= 0:
            raise ValueError("coverage_temperature must be positive")
        if not 0.0 <= exploration_prob <= 1.0:
            raise ValueError("exploration_prob must be between 0 and 1")
        if family_ids.ndim != 1 or family_ids.shape[0] != state.shape[0]:
            raise ValueError("family_ids must have one value per batch item")
        family_ids = family_ids.to(device=state.device, dtype=torch.long)
        if (family_ids < 0).any() or (family_ids >= self.family_count).any():
            raise ValueError("family_ids must identify valid families")

        batch = state.shape[0]
        family_sizes = torch.tensor(self.family_sizes, device=state.device,
                                    dtype=torch.long)
        family_offsets = torch.tensor(self.family_offsets, device=state.device,
                                      dtype=torch.long)
        selected_sizes = family_sizes[family_ids]
        selected_offsets = family_offsets[family_ids]
        routed_state = state + self.family_embeddings[family_ids]

        leaf = torch.zeros(batch, dtype=torch.long, device=state.device)
        entropies = []
        path_score = torch.zeros(batch, device=state.device)
        for level in range(self.local_depth):
            logits = (routed_state @ self.level_projections[0, level]
                      + self.level_bias[0, level])
            probs = F.softmax(logits, dim=-1)
            entropies.append(-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1))
            child = logits.argmax(dim=-1)
            if exploration_prob and self.training:
                explore = torch.rand(batch, device=state.device) < exploration_prob
                random_child = torch.randint(self.branch, (batch,), device=state.device)
                child = torch.where(explore, random_child, child)
            path_score = path_score + logits.gather(1, child.unsqueeze(1)).squeeze(1)
            leaf = leaf * self.branch + child

        local_base = leaf.remainder(selected_sizes)
        local_offsets = torch.arange(self.local_candidate_pool,
                                     device=state.device).view(1, -1)
        local_ids = (local_base.unsqueeze(-1) + local_offsets).remainder(
            selected_sizes.unsqueeze(-1)) + selected_offsets.unsqueeze(-1)
        shared_base = leaf.remainder(self.shared_count)
        shared_offsets = torch.arange(self.shared_candidate_pool,
                                      device=state.device).view(1, -1)
        shared_ids = (shared_base.unsqueeze(-1) + shared_offsets).remainder(
            self.shared_count)
        candidate_ids = torch.cat([local_ids, shared_ids], dim=-1)
        candidate_keys = self.keys[candidate_ids]
        candidate_logits = torch.einsum(
            "bd,bkd->bk", state, candidate_keys) / math.sqrt(state.shape[-1])
        top_values, top_positions = candidate_logits.topk(self.active_circuits, dim=-1)
        selected_ids = candidate_ids.gather(1, top_positions)
        weights = F.softmax(top_values, dim=-1)
        route_gain = 1.0 + 0.05 * torch.tanh(path_score)
        stats = {
            "router_entropy": torch.stack(entropies, dim=-1).mean(),
            "router_decisions": torch.tensor(self.local_depth, device=state.device),
            "route_gain": route_gain,
            "candidate_ids": candidate_ids,
            "selected_ids": selected_ids,
            "family_ids": family_ids,
            "family_sizes": selected_sizes,
            "shared_selected_fraction": selected_ids.lt(self.shared_count).float().mean(),
        }
        return selected_ids, weights, stats


class RoleAnchoredRouter(nn.Module):
    """Use a stable role tree for coarse routing and value state for scoring.

    Unlike ``StableFamilyRouter``, this router does not reserve private bank
    blocks for operators.  All rows remain in one physical bank.  A small
    role-only tree selects a coarse interleaved cell, then a value-dependent
    local tree and key score choose the active rows inside that cell.  The
    number and meaning of coarse cells are independent of total bank size, so
    extra capacity adds replicas to existing cells instead of changing the
    global competition seen by the role route.
    """

    def __init__(self, state_dim: int, num_circuits: int, branch: int = 8,
                 depth: int = 4, candidate_pool: int = 32,
                 active_circuits: int = 8, num_addresses: int = 1,
                 routing_capacity: int | None = None,
                 routing_depth: int | None = None, anchor_branch: int = 4,
                 anchor_depth: int = 2):
        super().__init__()
        if active_circuits > candidate_pool:
            raise ValueError("active_circuits cannot exceed candidate_pool")
        if num_addresses != 1:
            raise ValueError("RoleAnchoredRouter currently supports one address")
        if anchor_branch < 2 or anchor_depth < 1:
            raise ValueError("anchor_branch must be >=2 and anchor_depth must be positive")
        if anchor_branch ** anchor_depth >= num_circuits:
            raise ValueError("role anchor tree must have fewer cells than circuits")
        self.num_circuits = num_circuits
        self.branch = branch
        self.depth = depth
        self.candidate_pool = candidate_pool
        self.active_circuits = active_circuits
        self.num_addresses = num_addresses
        self.anchor_branch = anchor_branch
        self.anchor_depth = anchor_depth
        self.anchor_count = anchor_branch ** anchor_depth
        base_size, remainder = divmod(num_circuits, self.anchor_count)
        self.cell_sizes = tuple(
            base_size + (1 if index < remainder else 0)
            for index in range(self.anchor_count))
        if min(self.cell_sizes) < candidate_pool:
            raise ValueError("each role-anchor cell must fit the candidate pool")

        self.routing_capacity = (num_circuits if routing_capacity is None
                                 else int(routing_capacity))
        self.active_depth = depth if routing_depth is None else int(routing_depth)
        if self.routing_capacity != num_circuits:
            raise ValueError("RoleAnchoredRouter requires the full stable bank")
        if not 0 < self.active_depth <= depth:
            raise ValueError("routing_depth must be between 1 and depth")
        max_cell_size = max(self.cell_sizes)
        self.local_depth = min(self.active_depth,
                               max(1, math.ceil(math.log(max_cell_size, branch))))

        self.anchor_projections = nn.Parameter(
            torch.empty(anchor_depth, state_dim, anchor_branch))
        self.anchor_bias = nn.Parameter(torch.zeros(anchor_depth, anchor_branch))
        self.local_projections = nn.Parameter(
            torch.empty(depth, state_dim, branch))
        self.local_bias = nn.Parameter(torch.zeros(depth, branch))
        self.anchor_embeddings = nn.Parameter(torch.empty(self.anchor_count, state_dim))
        self.keys = nn.Parameter(torch.empty(num_circuits, state_dim))
        nn.init.normal_(self.anchor_projections, std=0.02)
        nn.init.normal_(self.local_projections, std=0.02)
        nn.init.normal_(self.anchor_embeddings, std=0.02)
        nn.init.normal_(self.keys, std=0.02)

    def set_routing_state(self, *, capacity: int | None = None,
                          depth: int | None = None) -> None:
        next_capacity = self.routing_capacity if capacity is None else int(capacity)
        next_depth = self.active_depth if depth is None else int(depth)
        if next_capacity != self.num_circuits:
            raise ValueError("RoleAnchoredRouter requires the full stable bank")
        if not 0 < next_depth <= self.depth:
            raise ValueError("routing_depth must be between 1 and depth")
        self.routing_capacity = next_capacity
        self.active_depth = next_depth
        self.local_depth = min(
            self.active_depth,
            max(1, math.ceil(math.log(max(self.cell_sizes), self.branch))))

    def forward(self, value_state: torch.Tensor, role_state: torch.Tensor,
                coverage: bool = False, coverage_temperature: float = 0.25,
                exploration_prob: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if coverage:
            raise NotImplementedError("RoleAnchoredRouter coverage is not implemented")
        if coverage_temperature <= 0:
            raise ValueError("coverage_temperature must be positive")
        if not 0.0 <= exploration_prob <= 1.0:
            raise ValueError("exploration_prob must be between 0 and 1")
        if value_state.shape != role_state.shape:
            raise ValueError("value_state and role_state must have the same shape")
        batch = value_state.shape[0]
        anchor_leaf = torch.zeros(batch, dtype=torch.long, device=value_state.device)
        anchor_entropies = []
        anchor_score = torch.zeros(batch, device=value_state.device)
        for level in range(self.anchor_depth):
            logits = (role_state @ self.anchor_projections[level]
                      + self.anchor_bias[level])
            probs = F.softmax(logits, dim=-1)
            anchor_entropies.append(
                -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1))
            child = logits.argmax(dim=-1)
            if exploration_prob and self.training:
                explore = torch.rand(batch, device=value_state.device) < exploration_prob
                random_child = torch.randint(self.anchor_branch, (batch,),
                                             device=value_state.device)
                child = torch.where(explore, random_child, child)
            anchor_score = anchor_score + logits.gather(
                1, child.unsqueeze(1)).squeeze(1)
            anchor_leaf = anchor_leaf * self.anchor_branch + child
        anchor_ids = anchor_leaf.remainder(self.anchor_count)

        cell_sizes = torch.tensor(self.cell_sizes, device=value_state.device,
                                  dtype=torch.long)
        selected_sizes = cell_sizes[anchor_ids]
        local_state = value_state + self.anchor_embeddings[anchor_ids]
        local_leaf = torch.zeros(batch, dtype=torch.long, device=value_state.device)
        local_entropies = []
        for level in range(self.local_depth):
            logits = (local_state @ self.local_projections[level]
                      + self.local_bias[level])
            probs = F.softmax(logits, dim=-1)
            local_entropies.append(
                -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1))
            child = logits.argmax(dim=-1)
            if exploration_prob and self.training:
                explore = torch.rand(batch, device=value_state.device) < exploration_prob
                random_child = torch.randint(self.branch, (batch,),
                                             device=value_state.device)
                child = torch.where(explore, random_child, child)
            local_leaf = local_leaf * self.branch + child

        local_base = local_leaf.remainder(selected_sizes)
        positions = torch.arange(self.candidate_pool,
                                 device=value_state.device).view(1, -1)
        local_positions = (local_base.unsqueeze(-1) + positions).remainder(
            selected_sizes.unsqueeze(-1))
        candidate_ids = anchor_ids.unsqueeze(-1) + local_positions * self.anchor_count
        candidate_keys = self.keys[candidate_ids]
        candidate_logits = torch.einsum(
            "bd,bkd->bk", value_state, candidate_keys) / math.sqrt(value_state.shape[-1])
        top_values, top_positions = candidate_logits.topk(self.active_circuits, dim=-1)
        selected_ids = candidate_ids.gather(1, top_positions)
        weights = F.softmax(top_values, dim=-1)
        route_gain = 1.0 + 0.05 * torch.tanh(anchor_score)
        stats = {
            "router_entropy": torch.stack(anchor_entropies + local_entropies,
                                           dim=-1).mean(),
            "router_decisions": torch.tensor(
                self.anchor_depth + self.local_depth, device=value_state.device),
            "route_gain": route_gain,
            "candidate_ids": candidate_ids,
            "selected_ids": selected_ids,
            "anchor_ids": anchor_ids,
            "cell_sizes": selected_sizes,
        }
        return selected_ids, weights, stats


class FixedRoleCellRouter(nn.Module):
    """Route value-dependent circuits inside stable operator/stage cells.

    Each role receives a deterministic interleaved cell, but every candidate
    in the pool remains local to that cell.  Unlike the learned role anchor,
    no coarse tree can collapse several roles onto one cell.  Unlike the
    earlier family-local prototype, the full candidate pool is available to
    value routing; there is no shared-fallback split that halves local width.
    """

    def __init__(self, state_dim: int, num_circuits: int, branch: int = 8,
                 depth: int = 4, candidate_pool: int = 32,
                 active_circuits: int = 8, num_addresses: int = 1,
                 routing_capacity: int | None = None,
                 routing_depth: int | None = None, role_count: int = 9):
        super().__init__()
        if active_circuits > candidate_pool:
            raise ValueError("active_circuits cannot exceed candidate_pool")
        if num_addresses != 1:
            raise ValueError("FixedRoleCellRouter currently supports one address")
        if role_count < 1 or role_count >= num_circuits:
            raise ValueError("role_count must be positive and smaller than num_circuits")
        self.num_circuits = num_circuits
        self.branch = branch
        self.depth = depth
        self.candidate_pool = candidate_pool
        self.active_circuits = active_circuits
        self.num_addresses = num_addresses
        self.role_count = role_count
        base_size, remainder = divmod(num_circuits, role_count)
        self.cell_sizes = tuple(
            base_size + (1 if index < remainder else 0)
            for index in range(role_count))
        if min(self.cell_sizes) < candidate_pool:
            raise ValueError("each role cell must fit the candidate pool")
        self.routing_capacity = (num_circuits if routing_capacity is None
                                 else int(routing_capacity))
        self.active_depth = depth if routing_depth is None else int(routing_depth)
        if self.routing_capacity != num_circuits:
            raise ValueError("FixedRoleCellRouter requires the full stable bank")
        if not 0 < self.active_depth <= depth:
            raise ValueError("routing_depth must be between 1 and depth")
        self.local_depth = min(
            self.active_depth,
            max(1, math.ceil(math.log(max(self.cell_sizes), branch))))
        self.local_projections = nn.Parameter(
            torch.empty(depth, state_dim, branch))
        self.local_bias = nn.Parameter(torch.zeros(depth, branch))
        self.role_embeddings = nn.Parameter(torch.empty(role_count, state_dim))
        self.keys = nn.Parameter(torch.empty(num_circuits, state_dim))
        nn.init.normal_(self.local_projections, std=0.02)
        nn.init.normal_(self.role_embeddings, std=0.02)
        nn.init.normal_(self.keys, std=0.02)

    def set_routing_state(self, *, capacity: int | None = None,
                          depth: int | None = None) -> None:
        next_capacity = self.routing_capacity if capacity is None else int(capacity)
        next_depth = self.active_depth if depth is None else int(depth)
        if next_capacity != self.num_circuits:
            raise ValueError("FixedRoleCellRouter requires the full stable bank")
        if not 0 < next_depth <= self.depth:
            raise ValueError("routing_depth must be between 1 and depth")
        self.routing_capacity = next_capacity
        self.active_depth = next_depth
        self.local_depth = min(
            self.active_depth,
            max(1, math.ceil(math.log(max(self.cell_sizes), self.branch))))

    def forward(self, value_state: torch.Tensor, role_ids: torch.Tensor,
                coverage: bool = False, coverage_temperature: float = 0.25,
                exploration_prob: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if coverage:
            raise NotImplementedError("FixedRoleCellRouter coverage is not implemented")
        if coverage_temperature <= 0:
            raise ValueError("coverage_temperature must be positive")
        if not 0.0 <= exploration_prob <= 1.0:
            raise ValueError("exploration_prob must be between 0 and 1")
        if role_ids.ndim != 1 or role_ids.shape[0] != value_state.shape[0]:
            raise ValueError("role_ids must have one value per batch item")
        role_ids = role_ids.to(device=value_state.device, dtype=torch.long)
        if (role_ids < 0).any() or (role_ids >= self.role_count).any():
            raise ValueError("role_ids must identify valid role cells")
        batch = value_state.shape[0]
        cell_sizes = torch.tensor(self.cell_sizes, device=value_state.device,
                                  dtype=torch.long)
        selected_sizes = cell_sizes[role_ids]
        local_state = value_state + self.role_embeddings[role_ids]
        local_leaf = torch.zeros(batch, dtype=torch.long, device=value_state.device)
        entropies = []
        path_score = torch.zeros(batch, device=value_state.device)
        for level in range(self.local_depth):
            logits = (local_state @ self.local_projections[level]
                      + self.local_bias[level])
            probs = F.softmax(logits, dim=-1)
            entropies.append(-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1))
            child = logits.argmax(dim=-1)
            if exploration_prob and self.training:
                explore = torch.rand(batch, device=value_state.device) < exploration_prob
                random_child = torch.randint(self.branch, (batch,),
                                             device=value_state.device)
                child = torch.where(explore, random_child, child)
            path_score = path_score + logits.gather(
                1, child.unsqueeze(1)).squeeze(1)
            local_leaf = local_leaf * self.branch + child
        local_base = local_leaf.remainder(selected_sizes)
        positions = torch.arange(self.candidate_pool,
                                 device=value_state.device).view(1, -1)
        local_positions = (local_base.unsqueeze(-1) + positions).remainder(
            selected_sizes.unsqueeze(-1))
        # Interleaving keeps role identity stable when rows are appended to a
        # larger bank: row r belongs to role r % role_count.
        candidate_ids = role_ids.unsqueeze(-1) + local_positions * self.role_count
        candidate_keys = self.keys[candidate_ids]
        candidate_logits = torch.einsum(
            "bd,bkd->bk", value_state, candidate_keys) / math.sqrt(value_state.shape[-1])
        top_values, top_positions = candidate_logits.topk(self.active_circuits, dim=-1)
        selected_ids = candidate_ids.gather(1, top_positions)
        weights = F.softmax(top_values, dim=-1)
        route_gain = 1.0 + 0.05 * torch.tanh(path_score)
        stats = {
            "router_entropy": torch.stack(entropies, dim=-1).mean(),
            "router_decisions": torch.tensor(self.local_depth, device=value_state.device),
            "route_gain": route_gain,
            "candidate_ids": candidate_ids,
            "selected_ids": selected_ids,
            "role_ids": role_ids,
            "cell_sizes": selected_sizes,
        }
        return selected_ids, weights, stats


class FactorizedRouter(nn.Module):
    """Address a virtual factorized bank through reusable factor keys.

    The ordinary router scores one key per virtual circuit.  That makes the
    routing problem grow linearly with stored capacity even when circuit
    parameters are factorized.  This router scores a single reusable factor
    table, forms a small Cartesian product of the best factor IDs, and then
    selects the active virtual circuit combinations.  Its route decision is
    therefore tied to factor semantics rather than to a growing global row
    index.
    """

    def __init__(self, state_dim: int, num_circuits: int, branch: int = 8,
                 depth: int = 4, candidate_pool: int = 32,
                 active_circuits: int = 8, num_addresses: int = 1,
                 routing_capacity: int | None = None,
                 routing_depth: int | None = None,
                 factor_count: int | None = None,
                 factor_candidate_pool: int | None = None,
                 operation_key_bank: bool = False):
        super().__init__()
        if num_addresses != 1:
            raise ValueError("FactorizedRouter currently supports one address")
        if active_circuits > candidate_pool:
            raise ValueError("active_circuits cannot exceed candidate_pool")
        if factor_count is None:
            factor_count = max(1, math.ceil(math.sqrt(num_circuits)))
        if factor_count < 1 or factor_count * factor_count < num_circuits:
            raise ValueError("factor_count must provide every virtual circuit ID")
        self.num_circuits = num_circuits
        self.branch = branch
        self.depth = depth
        self.candidate_pool = candidate_pool
        self.active_circuits = active_circuits
        self.num_addresses = num_addresses
        self.factor_count = factor_count
        self.operation_key_bank = bool(operation_key_bank)
        self.factor_candidate_pool = max(
            active_circuits,
            factor_candidate_pool or math.ceil(math.sqrt(candidate_pool)),
        )
        self.routing_capacity = (num_circuits if routing_capacity is None
                                 else int(routing_capacity))
        self.active_depth = depth if routing_depth is None else int(routing_depth)
        if not 0 < self.routing_capacity <= num_circuits:
            raise ValueError("routing_capacity must be between 1 and num_circuits")
        if not 0 < self.active_depth <= depth:
            raise ValueError("routing_depth must be between 1 and depth")
        self.keys = nn.Parameter(torch.empty(factor_count, state_dim))
        self.pair_interaction_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.normal_(self.keys, std=0.02)
        if self.operation_key_bank:
            # Start from the shared geometry and learn only an operation-local
            # residual, preserving transfer across primitives.
            self.operation_key_deltas = nn.Parameter(
                torch.zeros(3, factor_count, state_dim)
            )

    def set_routing_state(self, *, capacity: int | None = None,
                          depth: int | None = None) -> None:
        next_capacity = self.routing_capacity if capacity is None else int(capacity)
        next_depth = self.active_depth if depth is None else int(depth)
        if not 0 < next_capacity <= self.num_circuits:
            raise ValueError("routing capacity must be between 1 and num_circuits")
        if not 0 < next_depth <= self.depth:
            raise ValueError("routing depth must be between 1 and depth")
        self.routing_capacity = next_capacity
        self.active_depth = next_depth

    def _factor_ids(self, circuit_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = circuit_ids.remainder(self.factor_count)
        second = circuit_ids.div(self.factor_count, rounding_mode="floor")
        return first, second

    def forward(self, state: torch.Tensor, coverage: bool = False,
                coverage_temperature: float = 0.25,
                exploration_prob: float = 0.0,
                routing_offset: int | torch.Tensor = 0,
                routing_capacity: int | None = None,
                routing_windows: torch.Tensor | None = None,
                operation_ids: torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if coverage:
            raise NotImplementedError("FactorizedRouter coverage is not implemented")
        if coverage_temperature <= 0:
            raise ValueError("coverage_temperature must be positive")
        if not 0.0 <= exploration_prob <= 1.0:
            raise ValueError("exploration_prob must be between 0 and 1")
        if routing_windows is not None or (
                isinstance(routing_offset, int) and routing_offset != 0):
            raise ValueError("FactorizedRouter does not support bank windows")
        local_capacity = self.routing_capacity if routing_capacity is None else int(routing_capacity)
        if not 0 < local_capacity <= self.routing_capacity:
            raise ValueError("routing_capacity must be between 1 and the configured routing capacity")

        if self.operation_key_bank:
            if operation_ids is None:
                raise ValueError("operation_ids are required for operation key banks")
            if operation_ids.ndim != 1 or operation_ids.shape[0] != state.shape[0]:
                raise ValueError("operation_ids must have one value per batch item")
            operation_ids = operation_ids.to(device=state.device, dtype=torch.long)
            if (operation_ids < 0).any() or (operation_ids >= 3).any():
                raise ValueError("operation_ids must identify one of three operations")
            factor_keys = self.keys + self.operation_key_deltas[operation_ids]
            factor_logits = torch.einsum("bd,bfd->bf", state, factor_keys)
        else:
            factor_keys = self.keys
            factor_logits = torch.einsum("bd,fd->bf", state, factor_keys)
        factor_probs = F.softmax(factor_logits, dim=-1)
        entropy = -(factor_probs * factor_probs.clamp_min(1e-8).log()).sum(dim=-1)
        pool = min(self.factor_candidate_pool, self.factor_count)
        top_values, top_positions = factor_logits.topk(pool, dim=-1)
        first = top_positions.unsqueeze(-1).expand(-1, -1, pool)
        second = top_positions.unsqueeze(1).expand(-1, pool, -1)
        candidate_ids = first + self.factor_count * second
        if self.operation_key_bank:
            batch_ids = torch.arange(state.shape[0], device=state.device).view(-1, 1, 1)
            first_keys = factor_keys[batch_ids, first]
            second_keys = factor_keys[batch_ids, second]
        else:
            first_keys = self.keys[first]
            second_keys = self.keys[second]
        pair_interaction = torch.einsum(
            "bd,bijd->bij", state, first_keys * second_keys
        ) / math.sqrt(state.shape[-1])
        candidate_logits = (
            top_values.unsqueeze(-1) + top_values.unsqueeze(1)
            + self.pair_interaction_scale * pair_interaction
        )
        candidate_ids = candidate_ids.reshape(state.shape[0], -1)
        candidate_logits = candidate_logits.reshape(state.shape[0], -1)
        valid = candidate_ids < local_capacity
        candidate_logits = candidate_logits.masked_fill(~valid, torch.finfo(candidate_logits.dtype).min)
        candidate_pool = min(self.candidate_pool, candidate_ids.shape[1])
        top_values, top_positions = candidate_logits.topk(candidate_pool, dim=-1)
        selected_candidates = candidate_ids.gather(1, top_positions)
        top_values = top_values / math.sqrt(state.shape[-1])
        selected_ids = selected_candidates.gather(
            1, top_values.topk(self.active_circuits, dim=-1).indices)
        selected_values = top_values.gather(
            1, top_values.topk(self.active_circuits, dim=-1).indices)
        weights = F.softmax(selected_values, dim=-1)
        if exploration_prob and self.training:
            # Exploration stays within valid virtual addresses while preserving
            # the factorized address format.
            random_ids = torch.randint(local_capacity, selected_ids.shape,
                                       device=state.device)
            explore = torch.rand(state.shape[0], 1, device=state.device) < exploration_prob
            selected_ids = torch.where(explore, random_ids, selected_ids)
        stats = {
            "router_entropy": entropy,
            "router_decisions": torch.tensor(2, device=state.device),
            "route_gain": torch.ones(state.shape[0], device=state.device),
            "candidate_ids": selected_candidates,
            "selected_ids": selected_ids,
        }
        return selected_ids, weights, stats
