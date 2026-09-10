from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .factor_layout import build_factor_address_map


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
                 query_factor_mix_scale: float = 0.0,
                 factor_pair_rank: int = 0,
                 factor_pair_scale: float = 1.0,
                 factor_product_scale: float = 0.0,
                 factor_hidden_product_scale: float = 0.0,
                 factor_hidden_gate_scale: float = 0.0,
                 factor_composition_mode: str = "additive",
                 address_residual_rank: int = 0,
                 address_residual_scale: float = 1.0,
                 factor_address_layout: str = "standard",
                 legacy_factor_count: int | None = None):
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
        self.factor_pair_rank = int(factor_pair_rank)
        self.factor_pair_scale = float(factor_pair_scale)
        if self.factor_pair_rank < 0:
            raise ValueError("factor_pair_rank must be non-negative")
        if self.factor_pair_scale < 0.0:
            raise ValueError("factor_pair_scale must be non-negative")
        self.factor_product_scale = float(factor_product_scale)
        if self.factor_product_scale < 0.0:
            raise ValueError("factor_product_scale must be non-negative")
        self.factor_hidden_product_scale = float(factor_hidden_product_scale)
        if self.factor_hidden_product_scale < 0.0:
            raise ValueError("factor_hidden_product_scale must be non-negative")
        self.factor_hidden_gate_scale = float(factor_hidden_gate_scale)
        if self.factor_hidden_gate_scale < 0.0:
            raise ValueError("factor_hidden_gate_scale must be non-negative")
        if factor_composition_mode not in {"additive", "serial"}:
            raise ValueError("factor_composition_mode must be additive or serial")
        self.factor_composition_mode = factor_composition_mode
        self.factor_address_layout = factor_address_layout
        self.legacy_factor_count = legacy_factor_count
        self.address_residual_rank = int(address_residual_rank)
        self.address_residual_scale = float(address_residual_scale)
        if self.address_residual_rank < 0:
            raise ValueError("address_residual_rank must be non-negative")
        if self.address_residual_scale < 0.0:
            raise ValueError("address_residual_scale must be non-negative")
        factor_shape = (2, factor_count) if self.ordered_factor_slots else (factor_count,)
        self.down_factors = nn.Parameter(torch.empty(*factor_shape, state_dim, rank))
        self.up_factors = nn.Parameter(torch.empty(*factor_shape, rank, state_dim))
        self.bias_factors = nn.Parameter(torch.zeros(*factor_shape, state_dim))
        if self.query_factor_mix_scale:
            self.factor_gate_keys = nn.Parameter(
                torch.empty(*factor_shape, state_dim)
            )
            nn.init.normal_(self.factor_gate_keys, std=0.02)
        if self.factor_pair_rank:
            self.pair_codes = nn.Parameter(
                torch.empty(factor_count, self.factor_pair_rank)
            )
            self.pair_down_basis = nn.Parameter(
                torch.empty(self.factor_pair_rank, state_dim, rank)
            )
            self.pair_up_basis = nn.Parameter(
                torch.empty(self.factor_pair_rank, rank, state_dim)
            )
            self.pair_bias_basis = nn.Parameter(
                torch.empty(self.factor_pair_rank, state_dim)
            )
            nn.init.normal_(self.pair_codes, std=0.2)
            nn.init.normal_(self.pair_down_basis, std=0.02)
            nn.init.normal_(self.pair_up_basis, std=0.02)
            nn.init.normal_(self.pair_bias_basis, std=0.02)
        if self.address_residual_rank:
            self.address_residual_down = nn.Parameter(
                torch.empty(num_circuits, state_dim, self.address_residual_rank)
            )
            self.address_residual_up = nn.Parameter(
                torch.empty(num_circuits, self.address_residual_rank, state_dim)
            )
            self.address_residual_bias = nn.Parameter(
                torch.zeros(num_circuits, state_dim)
            )
            nn.init.normal_(self.address_residual_down, std=0.02)
            nn.init.normal_(self.address_residual_up, std=0.02)
        # A tiny per-address code preserves distinctions between combinations
        # without restoring a full independent matrix for every virtual row.
        mix_shape = (num_circuits, 2) if factor_mix_mode == "per_address" else (2,)
        self.factor_mix = nn.Parameter(torch.full(mix_shape, 0.5))
        self.factor_hidden_gates = nn.Parameter(torch.zeros(*factor_shape, rank))
        address_map = build_factor_address_map(
            num_circuits, factor_count, factor_address_layout, legacy_factor_count)
        self.register_buffer("_address_factor_ids", address_map, persistent=False)
        nn.init.normal_(self.down_factors, std=0.02)
        nn.init.normal_(self.up_factors, std=0.02)
        self.cache = None
        self.dispatch_backend = "torch"

    def set_cache(self, cache) -> None:
        # The existing CPU cache stores complete circuit rows and cannot be
        # used for factor composition. Keep the hook for shared interfaces;
        # factor rows remain resident and are gathered directly.
        self.cache = cache

    def _factor_ids(self, circuit_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._address_factor_ids is not None:
            factor_ids = self._address_factor_ids[circuit_ids]
            return factor_ids[..., 0], factor_ids[..., 1]
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
        if self.factor_pair_rank:
            pair_code = self.pair_codes[first] * self.pair_codes[second]
            down = down + self.factor_pair_scale * torch.einsum(
                "...p,pdr->...dr", pair_code, self.pair_down_basis
            )
            up = up + self.factor_pair_scale * torch.einsum(
                "...p,prd->...rd", pair_code, self.pair_up_basis
            )
            bias = bias + self.factor_pair_scale * torch.einsum(
                "...p,pd->...d", pair_code, self.pair_bias_basis
            )
        if self.factor_product_scale:
            down = down + self.factor_product_scale * first_down * second_down
            up = up + self.factor_product_scale * first_up * second_up
            bias = bias + self.factor_product_scale * first_bias * second_bias
        return down, up, bias

    def _gather_factor_slots(self, circuit_ids: torch.Tensor,
                             state: torch.Tensor | None = None):
        """Gather the two reusable factor paths for serial composition."""
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
        return (
            first_down * first_mix,
            first_up * first_mix,
            first_bias * first_bias_mix,
            second_down * second_mix,
            second_up * second_mix,
            second_bias * second_bias_mix,
        )

    def _gather_address_residual(self, circuit_ids: torch.Tensor):
        if not self.address_residual_rank:
            return None
        scale = self.address_residual_scale
        return (
            scale * self.address_residual_down[circuit_ids],
            scale * self.address_residual_up[circuit_ids],
            scale * self.address_residual_bias[circuit_ids],
        )

    def _gather_hidden_gate(self, circuit_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = self._factor_ids(circuit_ids)
        if self.ordered_factor_slots:
            return self.factor_hidden_gates[0, first], self.factor_hidden_gates[1, second]
        return self.factor_hidden_gates[first], self.factor_hidden_gates[second]

    def _native_fused_eligible(self, state: torch.Tensor) -> bool:
        """Return whether the inference-only CUDA kernel exactly covers this bank."""
        return (
            self.dispatch_backend == "native_cuda_fused"
            and state.device.type == "cuda"
            and state.dtype == torch.float32
            and not torch.is_grad_enabled()
            and self.ordered_factor_slots
            and self.factor_mix_mode == "per_address"
            and not self.query_factor_mix_scale
            and not self.factor_pair_rank
            and not self.factor_product_scale
            and not self.factor_hidden_product_scale
            and not self.factor_hidden_gate_scale
            and self.factor_composition_mode == "additive"
            and not self.address_residual_rank
        )

    def forward(self, state: torch.Tensor, circuit_ids: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        if self._native_fused_eligible(state):
            from .native_fused_dispatch import fused_factorized_dispatch
            return fused_factorized_dispatch(
                state, circuit_ids, weights, self.down_factors,
                self.up_factors, self.bias_factors, self.factor_mix,
                self._address_factor_ids,
            )
        if self.factor_composition_mode == "serial":
            first_down, first_up, first_bias, second_down, second_up, second_bias = (
                self._gather_factor_slots(circuit_ids, state)
            )
            first_hidden = F.gelu(torch.einsum("bd,bkdr->bkr", state, first_down))
            if self.factor_hidden_gate_scale:
                first_gate, second_gate = self._gather_hidden_gate(circuit_ids)
                first_hidden = first_hidden * (
                    1.0 + self.factor_hidden_gate_scale * torch.tanh(first_gate)
                )
            first_output = torch.einsum("bkr,bkrd->bkd", first_hidden, first_up) + first_bias
            middle = state.unsqueeze(1) + first_output
            second_hidden = F.gelu(torch.einsum("bkd,bkdr->bkr", middle, second_down))
            if self.factor_hidden_gate_scale:
                second_hidden = second_hidden * (
                    1.0 + self.factor_hidden_gate_scale * torch.tanh(second_gate)
                )
            outputs = torch.einsum("bkr,bkrd->bkd", second_hidden, second_up) + second_bias
            outputs = first_output + outputs
            if self.factor_hidden_product_scale:
                interaction_hidden = first_hidden * second_hidden
                interaction_up = 0.5 * (first_up + second_up)
                outputs = outputs + self.factor_hidden_product_scale * torch.einsum(
                    "bkr,bkrd->bkd", interaction_hidden, interaction_up
                )
            return (outputs * weights.unsqueeze(-1)).sum(dim=1)
        down, up, bias = self._gather(circuit_ids, state)
        hidden = torch.einsum("bd,bkdr->bkr", state, down)
        hidden = F.gelu(hidden)
        if self.factor_hidden_gate_scale:
            first_gate, second_gate = self._gather_hidden_gate(circuit_ids)
            hidden = hidden * (
                1.0 + self.factor_hidden_gate_scale * torch.tanh(first_gate + second_gate)
            )
        outputs = torch.einsum("bkr,bkrd->bkd", hidden, up) + bias
        if self.factor_hidden_product_scale:
            (first_down, first_up, _first_bias,
             second_down, second_up, _second_bias) = self._gather_factor_slots(
                circuit_ids, state
            )
            first_hidden = F.gelu(torch.einsum("bd,bkdr->bkr", state, first_down))
            second_hidden = F.gelu(torch.einsum("bd,bkdr->bkr", state, second_down))
            interaction_hidden = first_hidden * second_hidden
            interaction_up = 0.5 * (first_up + second_up)
            outputs = outputs + self.factor_hidden_product_scale * torch.einsum(
                "bkr,bkrd->bkd", interaction_hidden, interaction_up
            )
        residual = self._gather_address_residual(circuit_ids)
        if residual is not None:
            residual_down, residual_up, residual_bias = residual
            residual_hidden = torch.einsum(
                "bd,bkdr->bkr", state, residual_down
            )
            residual_hidden = F.gelu(residual_hidden)
            outputs = outputs + torch.einsum(
                "bkr,bkrd->bkd", residual_hidden, residual_up
            ) + residual_bias
        return (outputs * weights.unsqueeze(-1)).sum(dim=1)

    def forward_serial(self, state: torch.Tensor, circuit_ids: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
        current = state
        for slot in range(circuit_ids.shape[1]):
            if self.factor_composition_mode == "serial":
                (first_down, first_up, first_bias,
                 second_down, second_up, second_bias) = self._gather_factor_slots(
                    circuit_ids[:, slot], current
                )
                first_hidden = F.gelu(torch.einsum("bd,bdr->br", current, first_down))
                if self.factor_hidden_gate_scale:
                    first_gate, second_gate = self._gather_hidden_gate(circuit_ids[:, slot])
                    first_hidden = first_hidden * (
                        1.0 + self.factor_hidden_gate_scale * torch.tanh(first_gate)
                    )
                first_output = torch.einsum("br,brd->bd", first_hidden, first_up) + first_bias
                middle = current + first_output
                second_hidden = F.gelu(torch.einsum("bd,bdr->br", middle, second_down))
                if self.factor_hidden_gate_scale:
                    second_hidden = second_hidden * (
                        1.0 + self.factor_hidden_gate_scale * torch.tanh(second_gate)
                    )
                output = torch.einsum("br,brd->bd", second_hidden, second_up) + second_bias
                output = first_output + output
                if self.factor_hidden_product_scale:
                    interaction_hidden = first_hidden * second_hidden
                    interaction_up = 0.5 * (first_up + second_up)
                    output = output + self.factor_hidden_product_scale * torch.einsum(
                        "br,brd->bd", interaction_hidden, interaction_up
                    )
                current = current + weights[:, slot].unsqueeze(-1) * output
                continue
            down, up, bias = self._gather(circuit_ids[:, slot], current)
            hidden = torch.einsum("bd,bdr->br", current, down)
            hidden = F.gelu(hidden)
            if self.factor_hidden_gate_scale:
                first_gate, second_gate = self._gather_hidden_gate(circuit_ids[:, slot])
                hidden = hidden * (
                    1.0 + self.factor_hidden_gate_scale * torch.tanh(first_gate + second_gate)
                )
            output = torch.einsum("br,brd->bd", hidden, up) + bias
            if self.factor_hidden_product_scale:
                (first_down, first_up, _first_bias,
                 second_down, second_up, _second_bias) = self._gather_factor_slots(
                    circuit_ids[:, slot], current
                )
                first_hidden = F.gelu(torch.einsum("bd,bdr->br", current, first_down))
                second_hidden = F.gelu(torch.einsum("bd,bdr->br", current, second_down))
                interaction_hidden = first_hidden * second_hidden
                interaction_up = 0.5 * (first_up + second_up)
                output = output + self.factor_hidden_product_scale * torch.einsum(
                    "br,brd->bd", interaction_hidden, interaction_up
                )
            residual = self._gather_address_residual(circuit_ids[:, slot])
            if residual is not None:
                residual_down, residual_up, residual_bias = residual
                residual_hidden = torch.einsum(
                    "bd,bdr->br", current, residual_down
                )
                residual_hidden = F.gelu(residual_hidden)
                output = output + torch.einsum(
                    "br,brd->bd", residual_hidden, residual_up
                ) + residual_bias
            current = current + weights[:, slot].unsqueeze(-1) * output
        return current - state
