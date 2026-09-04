from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MesoscopicMacroCellBank(nn.Module):
    """Independent mesoscopic cells with an explicit state/memory interface.

    A selected cell applies one hidden nonlinear stage to a current state and
    a memory read.  The state and memory updates are then composed serially in
    the selected-slot order.  There is no inner router or attention path.
    """

    def __init__(self, num_cells: int, state_dim: int = 384,
                 hidden_dim: int = 480, bilinear_rank: int = 128,
                 residual_scale: float = 1.0) -> None:
        super().__init__()
        if min(num_cells, state_dim, hidden_dim, bilinear_rank) < 1:
            raise ValueError("cell dimensions must be positive")
        if residual_scale < 0.0:
            raise ValueError("residual_scale must be non-negative")
        self.num_cells = int(num_cells)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.bilinear_rank = int(bilinear_rank)
        self.residual_scale = float(residual_scale)

        self.state_down = nn.Parameter(torch.empty(num_cells, hidden_dim, state_dim))
        self.memory_down = nn.Parameter(torch.empty(num_cells, hidden_dim, state_dim))
        self.state_bilinear = nn.Parameter(torch.empty(num_cells, bilinear_rank, state_dim))
        self.memory_bilinear = nn.Parameter(torch.empty(num_cells, bilinear_rank, state_dim))
        self.bilinear_up = nn.Parameter(torch.empty(num_cells, hidden_dim, bilinear_rank))
        self.hidden_bias = nn.Parameter(torch.zeros(num_cells, hidden_dim))
        self.output_up = nn.Parameter(torch.empty(num_cells, state_dim, hidden_dim))
        self.memory_up = nn.Parameter(torch.empty(num_cells, state_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.state_down, std=1.0 / math.sqrt(self.state_dim))
        nn.init.normal_(self.memory_down, std=1.0 / math.sqrt(self.state_dim))
        nn.init.normal_(self.state_bilinear, std=1.0 / math.sqrt(self.state_dim))
        nn.init.normal_(self.memory_bilinear, std=1.0 / math.sqrt(self.state_dim))
        nn.init.normal_(self.bilinear_up, std=1.0 / math.sqrt(self.bilinear_rank))
        nn.init.normal_(self.output_up, std=1.0 / math.sqrt(self.hidden_dim))
        nn.init.normal_(self.memory_up, std=1.0 / math.sqrt(self.hidden_dim))

    @property
    def parameters_per_cell(self) -> int:
        return (
            4 * self.state_dim * self.hidden_dim
            + 2 * self.bilinear_rank * self.state_dim
            + self.hidden_dim * self.bilinear_rank
            + self.hidden_dim
        )

    @property
    def total_body_parameters(self) -> int:
        return self.num_cells * self.parameters_per_cell

    def _cell_update(self, state: torch.Tensor, memory: torch.Tensor,
                     cell_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        safe_ids = cell_ids.clamp_min(0)
        state_down = self.state_down[safe_ids]
        memory_down = self.memory_down[safe_ids]
        state_bilinear = self.state_bilinear[safe_ids]
        memory_bilinear = self.memory_bilinear[safe_ids]
        bilinear_up = self.bilinear_up[safe_ids]
        hidden_bias = self.hidden_bias[safe_ids]
        output_up = self.output_up[safe_ids]
        memory_up = self.memory_up[safe_ids]

        state_hidden = torch.einsum("bd,bhd->bh", state, state_down)
        memory_hidden = torch.einsum("bd,bhd->bh", memory, memory_down)
        state_code = torch.einsum("bd,bkd->bk", state, state_bilinear)
        memory_code = torch.einsum("bd,bkd->bk", memory, memory_bilinear)
        bilinear_hidden = torch.einsum(
            "bk,bhk->bh", state_code * memory_code, bilinear_up
        )
        hidden = F.gelu(state_hidden + memory_hidden + bilinear_hidden + hidden_bias)
        state_delta = torch.einsum("bh,bdh->bd", hidden, output_up)
        memory_delta = torch.einsum("bh,bdh->bd", hidden, memory_up)
        valid = cell_ids.ge(0).unsqueeze(-1)
        return torch.where(valid, state_delta, torch.zeros_like(state_delta)), torch.where(
            valid, memory_delta, torch.zeros_like(memory_delta)
        )

    def forward(self, state: torch.Tensor, memory: torch.Tensor,
                cell_ids: torch.Tensor, weights: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 2 or memory.shape != state.shape:
            raise ValueError("state and memory must have shape [batch, state_dim]")
        if cell_ids.ndim != 2 or weights.shape != cell_ids.shape:
            raise ValueError("cell_ids and weights must have shape [batch, slots]")
        if cell_ids.shape[0] != state.shape[0]:
            raise ValueError("state and cell_ids must have the same batch size")
        current_state = state
        current_memory = memory
        for slot in range(cell_ids.shape[1]):
            state_delta, memory_delta = self._cell_update(
                current_state, current_memory, cell_ids[:, slot]
            )
            scale = self.residual_scale * weights[:, slot].unsqueeze(-1)
            current_state = current_state + scale * state_delta
            current_memory = current_memory + scale * memory_delta
        return current_state, current_memory
