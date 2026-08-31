from __future__ import annotations

import torch
from torch import nn


class PersistentState(nn.Module):
    """Small recurrent controller used across internal reasoning steps."""

    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.initial = nn.Linear(input_dim, state_dim)
        self.update = nn.GRUCell(state_dim, state_dim)

    def initialize(self, encoded: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.initial(encoded))

    def step(self, state: torch.Tensor, circuit_delta: torch.Tensor) -> torch.Tensor:
        return self.update(circuit_delta, state)
