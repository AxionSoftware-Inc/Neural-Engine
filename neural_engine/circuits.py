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
                 factor_count: int | None = None,
                 factor_mix_mode: str = "per_address",
                 ordered_factor_slots: bool = False,
                 query_factor_mix_scale: float = 0.0):
        super().__init__()
        if num_circuits < 1:
            raise ValueError("num_circuits must be positive")
        if factor_count is None:
            factor_count = max(1, math.ceil(math.sqrt(num_circuits)))
        if factor_count < 1 or factor_count * factor_count < num_circuits:
            raise ValueError("factor_count must provide every virtual circuit ID")
        if factor_mix_mode not in {"per_address", "shared"}:
            raise ValueError("factor_mix_mode must be per_address or shared")
        self.num_circuits = num_circuits
        self.state_dim = state_dim
        self.rank = rank
        self.factor_count = factor_count
        self.factor_mix_mode = factor_mix_mode
        self.ordered_factor_slots = bool(ordered_factor_slots)
        self.query_factor_mix_scale = float(query_factor_mix_scale)
        if self.query_factor_mix_scale < 0.0:
            raise ValueError("query_factor_mix_scale must be non-negative")
        factor_shape = (2, factor_count) if self.ordered_factor_slots else (factor_count,)
        self.down_factors = nn.Parameter(torch.empty(*factor_shape, state_dim, rank))
        self.up_factors = nn.Parameter(torch.empty(*factor_shape, rank, state_dim))
        self.bias_factors = nn.Parameter(torch.zeros(*factor_shape, state_dim))
        if self.query_factor_mix_scale:
            self.factor_gate_keys = nn.Parameter(
                torch.empty(*factor_shape, state_dim)
            )
            nn.init.normal_(self.factor_gate_keys, std=0.02)
        # A tiny per-address code preserves distinctions between combinations
        # without restoring a full independent matrix for every virtual row.
        mix_shape = (num_circuits, 2) if factor_mix_mode == "per_address" else (2,)
        self.factor_mix = nn.Parameter(torch.full(mix_shape, 0.5))
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

    def _gather(self, circuit_ids: torch.Tensor,
                state: torch.Tensor | None = None):
        first, second = self._factor_ids(circuit_ids)
        if self.factor_mix_mode == "per_address":
            mix = self.factor_mix[circuit_ids]
            first_mix = mix[..., 0, None, None]
            second_mix = mix[..., 1, None, None]
            first_bias_mix = mix[..., 0, None]
            second_bias_mix = mix[..., 1, None]
        else:
            first_mix = self.factor_mix[0]
            second_mix = self.factor_mix[1]
            first_bias_mix = self.factor_mix[0]
            second_bias_mix = self.factor_mix[1]
        if self.query_factor_mix_scale:
            if state is None:
                raise ValueError("state is required for query-conditioned factor mixing")
            if self.ordered_factor_slots:
                first_gate_keys = self.factor_gate_keys[0, first]
                second_gate_keys = self.factor_gate_keys[1, second]
            else:
                first_gate_keys = self.factor_gate_keys[first]
                second_gate_keys = self.factor_gate_keys[second]
            if first_gate_keys.ndim == 2:
                first_score = torch.einsum("bd,bd->b", state, first_gate_keys)
                second_score = torch.einsum("bd,bd->b", state, second_gate_keys)
            else:
                first_score = torch.einsum("bd,bkd->bk", state, first_gate_keys)
                second_score = torch.einsum("bd,bkd->bk", state, second_gate_keys)
            first_gate = torch.tanh(first_score / math.sqrt(self.state_dim))
            second_gate = torch.tanh(second_score / math.sqrt(self.state_dim))
            first_mix = first_mix + self.query_factor_mix_scale * first_gate[..., None, None]
            second_mix = second_mix + self.query_factor_mix_scale * second_gate[..., None, None]
            first_bias_mix = first_bias_mix + self.query_factor_mix_scale * first_gate[..., None]
            second_bias_mix = second_bias_mix + self.query_factor_mix_scale * second_gate[..., None]
        if self.ordered_factor_slots:
            first_down = self.down_factors[0, first]
            second_down = self.down_factors[1, second]
            first_up = self.up_factors[0, first]
            second_up = self.up_factors[1, second]
            first_bias = self.bias_factors[0, first]
            second_bias = self.bias_factors[1, second]
        else:
            first_down = self.down_factors[first]
            second_down = self.down_factors[second]
            first_up = self.up_factors[first]
            second_up = self.up_factors[second]
            first_bias = self.bias_factors[first]
            second_bias = self.bias_factors[second]
        down = first_down * first_mix + second_down * second_mix
        up = first_up * first_mix + second_up * second_mix
        bias = first_bias * first_bias_mix + second_bias * second_bias_mix
        return down, up, bias

    def forward(self, state: torch.Tensor, circuit_ids: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        down, up, bias = self._gather(circuit_ids, state)
        hidden = torch.einsum("bd,bkdr->bkr", state, down)
        hidden = F.gelu(hidden)
        outputs = torch.einsum("bkr,bkrd->bkd", hidden, up) + bias
        return (outputs * weights.unsqueeze(-1)).sum(dim=1)

    def forward_serial(self, state: torch.Tensor, circuit_ids: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
        current = state
        for slot in range(circuit_ids.shape[1]):
            down, up, bias = self._gather(circuit_ids[:, slot], current)
            hidden = torch.einsum("bd,bdr->br", current, down)
            hidden = F.gelu(hidden)
            output = torch.einsum("br,brd->bd", hidden, up) + bias
            current = current + weights[:, slot].unsqueeze(-1) * output
        return current - state
