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
        self.has_gate_bias = bool(gate_bias)
        self.has_up_bias = bool(up_bias)
        self.has_down_bias = bool(down_bias)

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

    def _source_weights(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reassemble source Linear orientations for the exact all-active path."""
        if self.intermediate_size == self.num_circuits * self.chunk_size:
            gate_weight = self.gate_weight.reshape(self.intermediate_size, self.hidden_size)
            up_weight = self.up_weight.reshape(self.intermediate_size, self.hidden_size)
            down_weight = self.down_weight.permute(0, 2, 1).reshape(
                self.intermediate_size, self.hidden_size
            ).transpose(0, 1)
        else:
            widths = self.chunk_sizes.tolist()
            gate_weight = torch.cat([
                self.gate_weight[index, :width]
                for index, width in enumerate(widths)
            ], dim=0)
            up_weight = torch.cat([
                self.up_weight[index, :width]
                for index, width in enumerate(widths)
            ], dim=0)
            down_weight = torch.cat([
                self.down_weight[index, :, :width]
                for index, width in enumerate(widths)
            ], dim=1)
        return gate_weight, up_weight, down_weight

    def _source_biases(self) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Reassemble source bias vectors, preserving absent-bias semantics."""
        gate_bias = None
        up_bias = None
        if self.has_gate_bias:
            gate_bias = torch.cat([
                self.gate_bias[index, :width]
                for index, width in enumerate(self.chunk_sizes.tolist())
            ], dim=0)
        if self.has_up_bias:
            up_bias = torch.cat([
                self.up_bias[index, :width]
                for index, width in enumerate(self.chunk_sizes.tolist())
            ], dim=0)
        down_bias = self.down_bias if self.has_down_bias else None
        return gate_bias, up_bias, down_bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Evaluate every copied circuit; equal to the source gated FFN.

        The all-active path reassembles the three source Linear matrices before
        calling ``F.linear``. This preserves the source GEMM reduction order
        for strict pretrained-logit equivalence. Sparse execution uses
        ``forward_selected`` and intentionally avoids this reassembly.
        """
        flat, leading_shape = self._validate_hidden(hidden_states)
        gate_weight, up_weight, down_weight = self._source_weights()
        gate_bias, up_bias, down_bias = self._source_biases()
        gate = F.linear(flat, gate_weight, gate_bias)
        up = F.linear(flat, up_weight, up_bias)
        hidden = F.silu(gate) * up
        result = F.linear(hidden, down_weight, down_bias)
        return result.reshape(*leading_shape, self.hidden_size)

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


class LowRankResidual(nn.Module):
    """Small always-active residual used by teacher-distilled sparse pilots.

    The down projection is zero-initialized, so adding the residual cannot
    perturb an exact bank at initialization.  This gives a sparse route a
    cheap way to learn the aggregate effect of omitted circuits without
    reinstating the full FFN.
    """

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        if hidden_size < 1 or rank < 1:
            raise ValueError("hidden_size and rank must be positive")
        self.up = nn.Linear(hidden_size, rank, bias=False)
        self.down = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.down.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        parameter_dtype = self.up.weight.dtype
        values = self.up(hidden_states.to(parameter_dtype))
        result = self.down(F.silu(values))
        return result.to(hidden_states.dtype)


class TeacherDistilledSparseSwiGLU(nn.Module):
    """Soft-to-hard, teacher-distilled sparse execution for one FFN bank.

    During training, every frozen circuit is evaluated with a differentiable
    probability mass whose total is equal to the number of circuits.  This
    provides a global teacher-logit training signal without pretending that a
    hard top-k route is differentiable.  Evaluation switches to hard top-k
    and keeps the same mass-preserving normalization.  The router's final
    layer starts at zero, so the initial soft path is exactly the full bank.

    This is a research module, not a claim that dense scoring is a deployable
    router.  The training path is intentionally dense; deployment must replace
    it with a structured/cheap router after the quality gate passes.
    """

    def __init__(
        self,
        bank: SwiGLUCircuitBank,
        active_circuits: int,
        *,
        router_hidden: int = 128,
        residual_rank: int = 0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not 1 <= active_circuits <= bank.num_circuits:
            raise ValueError("active_circuits must be within the bank size")
        if router_hidden < 1:
            raise ValueError("router_hidden must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.bank = bank
        self.active_circuits = int(active_circuits)
        self.temperature = float(temperature)
        self.execution_mode = "soft"
        self.use_residual = True
        self.router = nn.Sequential(
            nn.Linear(bank.hidden_size, router_hidden),
            nn.SiLU(),
            nn.Linear(router_hidden, bank.num_circuits),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.residual = (
            LowRankResidual(bank.hidden_size, residual_rank)
            if residual_rank > 0 else None
        )

        # The copied teacher weights are not part of the pilot optimizer.
        for parameter in self.bank.parameters():
            parameter.requires_grad_(False)

    def set_execution_mode(self, mode: str) -> None:
        if mode not in {"soft", "straight_through", "hard"}:
            raise ValueError("execution mode must be soft, straight_through, or hard")
        self.execution_mode = mode

    def route_distribution(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return the differentiable full-bank route distribution."""
        router_dtype = self.router[0].weight.dtype
        scores = self.router(hidden_states.to(router_dtype))
        return F.softmax(scores / self.temperature, dim=-1)

    def _add_residual(
        self,
        hidden_states: torch.Tensor,
        result: torch.Tensor,
        use_residual: bool = True,
    ) -> torch.Tensor:
        if use_residual and self.use_residual and self.residual is not None:
            # The base Qwen stack is frozen; do not backpropagate through its
            # hidden-state producer just to train the small residual.
            result = result + self.residual(hidden_states.detach())
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.execution_mode in {"soft", "straight_through"}:
            router_dtype = self.router[0].weight.dtype
            scores = self.router(hidden_states.to(router_dtype))
            distribution = F.softmax(scores / self.temperature, dim=-1)
            # Sum of weights is N, making a uniform initial route identical
            # to the full bank output.
            weights = distribution * self.bank.num_circuits
            if self.execution_mode == "straight_through":
                _, top_ids = scores.topk(self.active_circuits, dim=-1)
                hard_distribution = torch.zeros_like(distribution)
                hard_distribution.scatter_(-1, top_ids, 1.0 / self.active_circuits)
                hard_weights = hard_distribution * self.bank.num_circuits
                # Forward uses exactly k circuits; backward follows the soft
                # distribution so the router can still receive global loss.
                weights = hard_weights + weights - weights.detach()
            outputs = self.bank.chunk_outputs(hidden_states.detach())
            result = (
                outputs * weights.unsqueeze(-1)
            ).sum(dim=-2) + self.bank.down_bias
            result = result.to(hidden_states.dtype)
            return self._add_residual(hidden_states, result)

        if self.active_circuits >= self.bank.num_circuits:
            return self._add_residual(hidden_states, self.bank(hidden_states.detach()))
        router_dtype = self.router[0].weight.dtype
        scores = self.router(hidden_states.to(router_dtype))
        top_values, top_ids = scores.topk(self.active_circuits, dim=-1)
        # Preserve the full-bank mass without allowing one selected circuit
        # to receive an unstable, score-dependent amplification.
        weights = torch.full_like(
            top_values, self.bank.num_circuits / self.active_circuits
        )
        result = self.bank.forward_selected(hidden_states.detach(), top_ids, weights)
        return self._add_residual(hidden_states, result)
