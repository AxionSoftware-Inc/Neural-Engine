from __future__ import annotations

import math

import torch
from torch import nn


class OperatorValuedLinear(nn.Module):
    """Packetized linear map with a shared learned operator basis.

    Each output-packet/input-packet block is

        Theta[o, i] = sum_a coeff[o, i, a] * basis[a]

    where every basis entry is a dense ``packet_width x packet_width`` matrix.
    The trainable scalar count is therefore explicit: ``q*g*g + E*q`` (plus
    bias), not the size of the materialized effective matrix.
    """

    def __init__(self, in_features: int, out_features: int,
                 packet_width: int = 16, basis_count: int = 8,
                 bias: bool = True) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")
        if packet_width < 1:
            raise ValueError("packet_width must be positive")
        if basis_count < 1:
            raise ValueError("basis_count must be positive")
        if in_features % packet_width or out_features % packet_width:
            raise ValueError("feature dimensions must be divisible by packet_width")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.packet_width = int(packet_width)
        self.basis_count = int(basis_count)
        self.input_packets = in_features // packet_width
        self.output_packets = out_features // packet_width
        self.basis = nn.Parameter(torch.empty(basis_count, packet_width, packet_width))
        self.coeff = nn.Parameter(torch.empty(
            self.output_packets, self.input_packets, basis_count
        ))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The scale keeps the summed operator span near a variance-preserving
        # linear map while leaving enough signal for the first update.
        nn.init.normal_(self.basis, std=1.0 / math.sqrt(self.packet_width))
        nn.init.normal_(self.coeff, std=1.0 / math.sqrt(self.basis_count))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def scalar_dof(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def effective_matrix_entries(self) -> int:
        return self.in_features * self.out_features

    @property
    def theoretical_macs(self) -> int:
        # Basis-first contraction: B_a x_i followed by coefficient mixing.
        basis_apply = self.input_packets * self.basis_count * self.packet_width**2
        mix = self.output_packets * self.input_packets * self.basis_count * self.packet_width
        return basis_apply + mix

    def effective_blocks(self) -> torch.Tensor:
        """Return blocks with shape [output_packet, input_packet, g, g]."""
        return torch.einsum("oia,agh->oigh", self.coeff, self.basis)

    def effective_weight(self) -> torch.Tensor:
        """Materialize the equivalent ordinary [out_features, in_features] map."""
        blocks = self.effective_blocks()
        return blocks.permute(0, 3, 1, 2).reshape(self.out_features, self.in_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dimension {self.in_features}, got {inputs.shape[-1]}"
            )
        leading_shape = inputs.shape[:-1]
        packets = inputs.reshape(-1, self.input_packets, self.packet_width)
        transformed = torch.einsum("nig,agh->niah", packets, self.basis)
        outputs = torch.einsum("oia,niah->noh", self.coeff, transformed)
        outputs = outputs.reshape(*leading_shape, self.out_features)
        if self.bias is not None:
            outputs = outputs + self.bias
        return outputs

    def parameter_report(self) -> dict[str, int | float]:
        scalar_dof = self.scalar_dof
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "packet_width": self.packet_width,
            "basis_count": self.basis_count,
            "input_packets": self.input_packets,
            "output_packets": self.output_packets,
            "scalar_dof": scalar_dof,
            "effective_matrix_entries": self.effective_matrix_entries,
            "compression_ratio": self.effective_matrix_entries / max(scalar_dof, 1),
            "theoretical_macs": self.theoretical_macs,
        }
