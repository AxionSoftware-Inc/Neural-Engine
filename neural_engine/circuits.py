from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MicroCircuitBank(nn.Module):
    """Many small low-rank blocks stored as contiguous parameter tensors."""

    def __init__(self, num_circuits: int, state_dim: int, rank: int):
        super().__init__()
        self.num_circuits = num_circuits
        self.state_dim = state_dim
        self.rank = rank
        self.down = nn.Parameter(torch.empty(num_circuits, state_dim, rank))
        self.up = nn.Parameter(torch.empty(num_circuits, rank, state_dim))
        self.bias = nn.Parameter(torch.zeros(num_circuits, state_dim))
        nn.init.normal_(self.down, std=0.02)
        nn.init.normal_(self.up, std=0.02)

    def forward(self, state: torch.Tensor, circuit_ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        down = self.down[circuit_ids]
        up = self.up[circuit_ids]
        bias = self.bias[circuit_ids]
        hidden = torch.einsum("bd,bkdr->bkr", state, down)
        hidden = F.gelu(hidden)
        outputs = torch.einsum("bkr,bkrd->bkd", hidden, up) + bias
        return (outputs * weights.unsqueeze(-1)).sum(dim=1)
