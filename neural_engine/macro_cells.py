from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .instrumentation import count_parameters


class MacroCellBank(nn.Module):
    """Sparse bank of reusable multi-step register transformations.

    A macro-cell is a short serial program of low-rank primitive transforms.
    The router selects one or a few macro-cell IDs, and only those rows are
    gathered for the current register update.  The bank can therefore grow in
    stored capacity without making every macro-cell active on every sample.

    This module deliberately has no attention or global all-bank operation.
    Each selected macro-cell owns a small local sequence and communicates with
    the rest of the engine only through the shared state vector.
    """

    def __init__(
        self,
        num_cells: int,
        state_dim: int,
        rank: int = 8,
        depth: int = 4,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if num_cells < 1:
            raise ValueError("num_cells must be positive")
        if state_dim < 1 or rank < 1 or depth < 1:
            raise ValueError("state_dim, rank, and depth must be positive")
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")
        self.num_cells = int(num_cells)
        self.state_dim = int(state_dim)
        self.rank = int(rank)
        self.depth = int(depth)
        self.residual_scale = float(residual_scale)

        self.down = nn.Parameter(
            torch.empty(num_cells, depth, state_dim, rank)
        )
        self.up = nn.Parameter(
            torch.empty(num_cells, depth, rank, state_dim)
        )
        self.bias = nn.Parameter(torch.zeros(num_cells, depth, state_dim))
        nn.init.normal_(self.down, std=0.02)
        nn.init.normal_(self.up, std=0.02)

    @property
    def parameters_per_cell(self) -> int:
        return self.depth * (
            self.state_dim * self.rank
            + self.rank * self.state_dim
            + self.state_dim
        )

    def forward(
        self,
        state: torch.Tensor,
        cell_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Apply selected macro-cells and return only their residual delta."""
        if state.ndim != 2:
            raise ValueError("state must have shape [batch, state_dim]")
        if cell_ids.ndim != 2 or weights.shape != cell_ids.shape:
            raise ValueError("cell_ids and weights must have shape [batch, slots]")
        if cell_ids.shape[0] != state.shape[0]:
            raise ValueError("state and cell_ids must have the same batch size")

        current = state
        for slot in range(cell_ids.shape[1]):
            ids = cell_ids[:, slot]
            valid = ids.ge(0)
            safe_ids = ids.clamp_min(0)
            down = self.down[safe_ids]
            up = self.up[safe_ids]
            bias = self.bias[safe_ids]
            slot_state = current
            slot_weight = weights[:, slot].unsqueeze(-1)
            for layer in range(self.depth):
                hidden = torch.einsum("bd,bdr->br", slot_state, down[:, layer])
                hidden = F.gelu(hidden)
                delta = torch.einsum(
                    "br,brd->bd", hidden, up[:, layer]
                ) + bias[:, layer]
                updated = slot_state + (
                    self.residual_scale * slot_weight * delta
                )
                slot_state = torch.where(valid.unsqueeze(-1), updated, slot_state)
            current = slot_state
        return current - state

    def parameter_report(self) -> dict[str, int | float | bool]:
        total = count_parameters(self)
        return {
            "total_params": total,
            "parameters_per_cell": self.parameters_per_cell,
            "num_cells": self.num_cells,
            "rank": self.rank,
            "depth": self.depth,
            "residual_scale": self.residual_scale,
            "attention": False,
        }
