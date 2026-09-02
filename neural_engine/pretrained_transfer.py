"""Exact conversion utilities for pretrained gated FFN layers.

Qwen3 uses the common SwiGLU form::

    down_proj(silu(gate_proj(x)) * up_proj(x))

This module splits the intermediate dimension into contiguous circuit chunks.
With every chunk active, the circuit bank is algebraically identical to the
source FFN.  It is intentionally separate from the learned Neural Engine
router: exact conversion is the required gate before sparse routing or
factor-sharing is attempted.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLUCircuitBank(nn.Module):
    """Contiguous SwiGLU slices copied from three ``nn.Linear`` projections.

    A circuit is one contiguous slice of the source intermediate dimension.
    The final circuit may be padded when ``intermediate_size`` is not a
    multiple of ``chunk_size``; padded rows are masked and initialized to
    zero.  ``forward`` evaluates all circuits and is the exact-conversion
    path.  ``forward_selected`` evaluates only supplied circuit IDs and is
    used for later sparse-routing experiments.

    The bank stores source Linear weights in PyTorch's native orientation:
    ``gate/up`` are ``[intermediate, hidden]`` and ``down`` is
    ``[hidden, intermediate]``.  Qwen projections normally have no bias, but
    optional biases are supported so the conversion test also covers generic
    gated FFN layers.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        chunk_size: int,
        *,
        gate_bias: bool = False,
        up_bias: bool = False,
        down_bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or intermediate_size < 1 or chunk_size < 1:
            raise ValueError("hidden_size, intermediate_size, and chunk_size must be positive")
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.chunk_size = int(chunk_size)
        self.num_circuits = math.ceil(self.intermediate_size / self.chunk_size)

        self.gate_weight = nn.Parameter(
            torch.empty(self.num_circuits, self.chunk_size, self.hidden_size)
        )
        self.up_weight = nn.Parameter(
            torch.empty(self.num_circuits, self.chunk_size, self.hidden_size)
        )
        self.down_weight = nn.Parameter(
            torch.empty(self.num_circuits, self.hidden_size, self.chunk_size)
        )

        chunk_sizes = [
            min(self.chunk_size, self.intermediate_size - start)
            for start in range(0, self.intermediate_size, self.chunk_size)
        ]
        self.register_buffer("chunk_sizes", torch.tensor(chunk_sizes, dtype=torch.long))
        self.register_buffer(
            "valid_mask",
            torch.arange(self.chunk_size).unsqueeze(0) < self.chunk_sizes.unsqueeze(1),
        )

        if gate_bias:
            self.gate_bias = nn.Parameter(torch.zeros(self.num_circuits, self.chunk_size))
        else:
            self.register_buffer("gate_bias", torch.zeros(self.num_circuits, self.chunk_size))
        if up_bias:
            self.up_bias = nn.Parameter(torch.zeros(self.num_circuits, self.chunk_size))
        else:
            self.register_buffer("up_bias", torch.zeros(self.num_circuits, self.chunk_size))
        if down_bias:
            self.down_bias = nn.Parameter(torch.zeros(self.hidden_size))
        else:
            self.register_buffer("down_bias", torch.zeros(self.hidden_size))

        nn.init.normal_(self.gate_weight, std=0.02)
        nn.init.normal_(self.up_weight, std=0.02)
        nn.init.normal_(self.down_weight, std=0.02)
        self._zero_padding()

    @classmethod
    def from_linear_layers(
        cls,
        gate_proj: nn.Linear,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
        chunk_size: int,
    ) -> "SwiGLUCircuitBank":
        """Copy a gated FFN made from ``gate_proj``, ``up_proj``, ``down_proj``.

        This matches Qwen-style modules whose projections are named
        ``gate_proj``, ``up_proj``, and ``down_proj``.  No source parameter is
        shared with the bank; the returned module is an independent copy.
        """
        if not all(isinstance(layer, nn.Linear) for layer in (gate_proj, up_proj, down_proj)):
            raise TypeError("all projections must be torch.nn.Linear instances")
        hidden_size = gate_proj.in_features
        intermediate_size = gate_proj.out_features
        if up_proj.in_features != hidden_size or up_proj.out_features != intermediate_size:
            raise ValueError("gate_proj and up_proj must have matching dimensions")
        if down_proj.in_features != intermediate_size or down_proj.out_features != hidden_size:
            raise ValueError("down_proj must map intermediate_size back to hidden_size")
        if gate_proj.weight.device != up_proj.weight.device or gate_proj.weight.device != down_proj.weight.device:
            raise ValueError("all projections must be on the same device")
        bank = cls(
            hidden_size,
            intermediate_size,
            chunk_size,
            gate_bias=gate_proj.bias is not None,
            up_bias=up_proj.bias is not None,
            down_bias=down_proj.bias is not None,
        ).to(device=gate_proj.weight.device, dtype=gate_proj.weight.dtype)
        with torch.no_grad():
            bank.gate_weight.zero_()
            bank.up_weight.zero_()
            bank.down_weight.zero_()
            for circuit_id, start in enumerate(range(0, intermediate_size, chunk_size)):
                end = min(start + chunk_size, intermediate_size)
                width = end - start
                bank.gate_weight[circuit_id, :width].copy_(gate_proj.weight[start:end])
                bank.up_weight[circuit_id, :width].copy_(up_proj.weight[start:end])
                bank.down_weight[circuit_id, :, :width].copy_(down_proj.weight[:, start:end])
                if gate_proj.bias is not None:
                    bank.gate_bias[circuit_id, :width].copy_(gate_proj.bias[start:end])
                if up_proj.bias is not None:
                    bank.up_bias[circuit_id, :width].copy_(up_proj.bias[start:end])
            if down_proj.bias is not None:
                bank.down_bias.copy_(down_proj.bias)
            bank._zero_padding()
        return bank

    @classmethod
    def from_qwen_mlp(cls, mlp: nn.Module, chunk_size: int) -> "SwiGLUCircuitBank":
        """Convert a Qwen-style MLP exposing the three projection attributes."""
        try:
            gate_proj = mlp.gate_proj
            up_proj = mlp.up_proj
            down_proj = mlp.down_proj
        except AttributeError as exc:
            raise TypeError("the MLP must expose gate_proj, up_proj, and down_proj") from exc
        return cls.from_linear_layers(gate_proj, up_proj, down_proj, chunk_size)

    def _zero_padding(self) -> None:
        """Keep invalid rows inert, including after a source-weight copy."""
        with torch.no_grad():
            self.gate_weight.mul_(self.valid_mask.unsqueeze(-1))
            self.up_weight.mul_(self.valid_mask.unsqueeze(-1))
            self.down_weight.mul_(self.valid_mask.unsqueeze(1))
            self.gate_bias.mul_(self.valid_mask)
            self.up_bias.mul_(self.valid_mask)

    def _validate_hidden(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"last dimension must be {self.hidden_size}, got {hidden_states.shape[-1]}"
            )
        return hidden_states.reshape(-1, self.hidden_size), tuple(hidden_states.shape[:-1])

    def chunk_outputs(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return each circuit contribution, excluding the shared down bias.

        Shape is ``[*hidden_states.shape[:-1], num_circuits, hidden_size]``.
        This is useful for offline routing analysis.  Computing it evaluates
        every circuit, so a router must not use this helper in a deployment
        path; it is an oracle/diagnostic only.
        """
        flat, leading_shape = self._validate_hidden(hidden_states)
        gate = torch.einsum("nh,ckh->nck", flat, self.gate_weight)
        up = torch.einsum("nh,ckh->nck", flat, self.up_weight)
        gate = gate + self.gate_bias.unsqueeze(0)
        up = up + self.up_bias.unsqueeze(0)
        hidden = F.silu(gate) * up
        hidden = hidden * self.valid_mask.unsqueeze(0).to(hidden.dtype)
        outputs = torch.einsum("nck,chk->nch", hidden, self.down_weight)
        return outputs.reshape(*leading_shape, self.num_circuits, self.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Evaluate every copied circuit; equal to the source gated FFN."""
        outputs = self.chunk_outputs(hidden_states)
        return outputs.sum(dim=-2) + self.down_bias

    def forward_selected(
        self,
        hidden_states: torch.Tensor,
        circuit_ids: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate only selected circuits and add the source down bias once.

        ``circuit_ids`` must have shape ``[batch, slots]`` for a 2-D input or
        the corresponding leading shape plus ``slots``.  This additive form
        makes all-circuit selection with unit weights exactly reproduce
        ``forward``.  Sparse approximations can use normalized or learned
        weights, but the shared bias is still added once.
        """
        flat, leading_shape = self._validate_hidden(hidden_states)
        ids = circuit_ids.to(device=flat.device, dtype=torch.long)
        if tuple(ids.shape[:-1]) != leading_shape:
            raise ValueError(
                f"circuit_ids leading shape must be {leading_shape}, got {tuple(ids.shape[:-1])}"
            )
        ids = ids.reshape(-1, ids.shape[-1])
        if ids.shape[-1] < 1:
            raise ValueError("circuit_ids must contain at least one slot")
        if bool((ids < 0).any()) or bool((ids >= self.num_circuits).any()):
            raise ValueError("circuit_ids contains an invalid circuit address")
        if weights is None:
            route_weights = torch.ones(ids.shape, device=flat.device, dtype=flat.dtype)
        else:
            route_weights = weights.to(device=flat.device, dtype=flat.dtype)
            if tuple(route_weights.shape) != tuple(circuit_ids.shape):
                raise ValueError("weights must have the same shape as circuit_ids")
            route_weights = route_weights.reshape_as(ids)

        gate_weight = self.gate_weight[ids]
        up_weight = self.up_weight[ids]
        down_weight = self.down_weight[ids]
        gate_bias = self.gate_bias[ids]
        up_bias = self.up_bias[ids]
        valid_mask = self.valid_mask[ids]
        gate = torch.einsum("nh,nqkh->nqk", flat, gate_weight) + gate_bias
        up = torch.einsum("nh,nqkh->nqk", flat, up_weight) + up_bias
        hidden = F.silu(gate) * up
        hidden = hidden * valid_mask.to(hidden.dtype)
        outputs = torch.einsum("nqk,nqhk->nqh", hidden, down_weight)
        result = (outputs * route_weights.unsqueeze(-1)).sum(dim=1) + self.down_bias
        return result.reshape(*leading_shape, self.hidden_size)

    def parameter_report(self) -> dict[str, int | float]:
        """Report copied capacity and the active cost of a selected circuit."""
        trainable = sum(parameter.numel() for parameter in self.parameters())
        per_circuit = self.chunk_size * self.hidden_size * 3
        return {
            "num_circuits": self.num_circuits,
            "intermediate_size": self.intermediate_size,
            "chunk_size": self.chunk_size,
            "total_parameters": int(trainable),
            "active_parameters_per_circuit": int(per_circuit),
            "active_fraction_one_circuit": float(per_circuit / max(trainable, 1)),
        }


def top_contribution_circuits(
    bank: SwiGLUCircuitBank,
    hidden_states: torch.Tensor,
    active_circuits: int,
) -> torch.Tensor:
    """Return an offline top-contribution route for an upper-bound audit.

    The function intentionally computes all circuit outputs first.  It is not
    a cheap router and must not be treated as evidence of deployable sparse
    inference; it only answers how much error remains if the most useful
    chunks were known in advance.
    """
    if active_circuits < 1 or active_circuits > bank.num_circuits:
        raise ValueError("active_circuits must be within the circuit-bank size")
    scores = bank.chunk_outputs(hidden_states).norm(dim=-1)
    return scores.topk(active_circuits, dim=-1).indices
