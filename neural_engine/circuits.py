from __future__ import annotations

import math

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
        self.cache = None

    def set_cache(self, cache) -> None:
        self.cache = cache

    def forward(self, state: torch.Tensor, circuit_ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if self.cache is None:
            down = self.down[circuit_ids]
            up = self.up[circuit_ids]
            bias = self.bias[circuit_ids]
        else:
            down, up, bias = self.cache.gather(circuit_ids)
        hidden = torch.einsum("bd,bkdr->bkr", state, down)
        hidden = F.gelu(hidden)
        outputs = torch.einsum("bkr,bkrd->bkd", hidden, up) + bias
        return (outputs * weights.unsqueeze(-1)).sum(dim=1)

    def forward_serial(self, state: torch.Tensor, circuit_ids: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
        """Compose the selected circuits in score order instead of mixing them.

        The same active circuit budget is preserved. Each selected block sees
        the residual produced by the previous block, which gives the routed
        bank an explicit compositional execution mode.
        """
        current = state
        for slot in range(circuit_ids.shape[1]):
            if self.cache is None:
                down = self.down[circuit_ids[:, slot]]
                up = self.up[circuit_ids[:, slot]]
                bias = self.bias[circuit_ids[:, slot]]
            else:
                down, up, bias = self.cache.gather(circuit_ids[:, slot])
            hidden = torch.einsum("bd,bdr->br", current, down)
            hidden = F.gelu(hidden)
            output = torch.einsum("br,brd->bd", hidden, up) + bias
            current = current + weights[:, slot].unsqueeze(-1) * output
        return current - state


class SharedResidualMicroCircuitBank(MicroCircuitBank):
    """Independent sparse circuits with one shared low-rank residual path.

    The shared path gives every routed trajectory a common primitive transform
    and receives gradient on every sample.  The per-circuit rows remain sparse
    adapters, so stored capacity can grow without making the whole bank dense.
    """

    def __init__(self, num_circuits: int, state_dim: int, rank: int,
                 shared_rank: int = 8):
        super().__init__(num_circuits, state_dim, rank)
        if shared_rank < 1:
            raise ValueError("shared_rank must be positive")
        self.shared_rank = shared_rank
        self.shared_down = nn.Parameter(torch.empty(state_dim, shared_rank))
        self.shared_up = nn.Parameter(torch.empty(shared_rank, state_dim))
        self.shared_bias = nn.Parameter(torch.zeros(state_dim))
        nn.init.normal_(self.shared_down, std=0.02)
        nn.init.normal_(self.shared_up, std=0.02)

    def _shared_forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = torch.einsum("bd,dr->br", state, self.shared_down)
        hidden = F.gelu(hidden)
        return torch.einsum("br,rd->bd", hidden, self.shared_up) + self.shared_bias

    def forward(self, state: torch.Tensor, circuit_ids: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        return self._shared_forward(state) + super().forward(state, circuit_ids, weights)

    def forward_serial(self, state: torch.Tensor, circuit_ids: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
        current = state + self._shared_forward(state)
        for slot in range(circuit_ids.shape[1]):
            if self.cache is None:
                down = self.down[circuit_ids[:, slot]]
                up = self.up[circuit_ids[:, slot]]
                bias = self.bias[circuit_ids[:, slot]]
            else:
                down, up, bias = self.cache.gather(circuit_ids[:, slot])
            hidden = torch.einsum("bd,bdr->br", current, down)
            hidden = F.gelu(hidden)
            output = torch.einsum("br,brd->bd", hidden, up) + bias
            current = current + weights[:, slot].unsqueeze(-1) * output
        return current - state


class FactorizedMicroCircuitBank(nn.Module):
    """Virtual circuit bank composed from reusable factor rows.

    A virtual circuit ID is represented by two factor IDs.  Its low-rank
    matrices and bias are the learned weighted sum of those factor rows.  The
    virtual bank can therefore expose ``factor_count ** 2`` addresses while
    each selected route touches only two rows from each factor table.  Factor
    rows are shared by many virtual addresses, which gives larger banks a
    reusable gradient path instead of one independently trained island per
    address.
    """

    def __init__(self, num_circuits: int, state_dim: int, rank: int,
                 factor_count: int | None = None):
        super().__init__()
        if num_circuits < 1:
            raise ValueError("num_circuits must be positive")
        if factor_count is None:
            factor_count = max(1, math.ceil(math.sqrt(num_circuits)))
        if factor_count < 1 or factor_count * factor_count < num_circuits:
            raise ValueError("factor_count must provide every virtual circuit ID")
        self.num_circuits = num_circuits
        self.state_dim = state_dim
        self.rank = rank
        self.factor_count = factor_count
        self.down_factors = nn.Parameter(torch.empty(factor_count, state_dim, rank))
        self.up_factors = nn.Parameter(torch.empty(factor_count, rank, state_dim))
        self.bias_factors = nn.Parameter(torch.zeros(factor_count, state_dim))
        # A tiny per-address code preserves distinctions between combinations
        # without restoring a full independent matrix for every virtual row.
        self.factor_mix = nn.Parameter(torch.full((num_circuits, 2), 0.5))
        nn.init.normal_(self.down_factors, std=0.02)
        nn.init.normal_(self.up_factors, std=0.02)
        self.cache = None

    def set_cache(self, cache) -> None:
        # The existing CPU cache stores complete circuit rows and cannot be
        # used for factor composition. Keep the hook for shared interfaces;
        # factor rows remain resident and are gathered directly.
        self.cache = cache

    def _factor_ids(self, circuit_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = circuit_ids.remainder(self.factor_count)
        second = circuit_ids.div(self.factor_count, rounding_mode="floor")
        return first, second

    def _gather(self, circuit_ids: torch.Tensor):
        first, second = self._factor_ids(circuit_ids)
        mix = self.factor_mix[circuit_ids]
        down = (self.down_factors[first] * mix[..., 0, None, None]
                + self.down_factors[second] * mix[..., 1, None, None])
        up = (self.up_factors[first] * mix[..., 0, None, None]
              + self.up_factors[second] * mix[..., 1, None, None])
        bias = (self.bias_factors[first] * mix[..., 0, None]
                + self.bias_factors[second] * mix[..., 1, None])
        return down, up, bias

    def forward(self, state: torch.Tensor, circuit_ids: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        down, up, bias = self._gather(circuit_ids)
        hidden = torch.einsum("bd,bkdr->bkr", state, down)
        hidden = F.gelu(hidden)
        outputs = torch.einsum("bkr,bkrd->bkd", hidden, up) + bias
        return (outputs * weights.unsqueeze(-1)).sum(dim=1)

    def forward_serial(self, state: torch.Tensor, circuit_ids: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
        current = state
        for slot in range(circuit_ids.shape[1]):
            down, up, bias = self._gather(circuit_ids[:, slot])
            hidden = torch.einsum("bd,bdr->br", current, down)
            hidden = F.gelu(hidden)
            output = torch.einsum("br,brd->bd", hidden, up) + bias
            current = current + weights[:, slot].unsqueeze(-1) * output
        return current - state
