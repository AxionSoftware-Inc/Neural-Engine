from __future__ import annotations

import math

import torch
from torch import nn


VALUE_TOKEN_OFFSET = 32
VALUE_MODULUS = 64
VALUE_HARMONICS = (1, 2, 4, 8, 16, 32)


class FixedFourierValueEncoder(nn.Module):
    """Keep modular Fourier coordinates separate instead of learning a mix."""

    def __init__(self, output_dim: int, include_linear: bool = False):
        super().__init__()
        feature_count = 1 + 2 * len(VALUE_HARMONICS)
        start = 0 if include_linear else 1
        source = torch.arange(start, feature_count, dtype=torch.long)
        indices = source.repeat((output_dim + source.numel() - 1) // source.numel())
        self.register_buffer("indices", indices[:output_dim], persistent=False)
        self.scale = math.sqrt(source.numel() / output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.index_select(-1, self.indices) * self.scale


class HybridFourierValueEncoder(nn.Module):
    """Combine adaptable value features with an unmixed modular basis."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.learned = nn.Linear(1 + 2 * len(VALUE_HARMONICS), output_dim)
        self.fixed = FixedFourierValueEncoder(output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.learned(features) + self.fixed(features)


def encode_tokens(inputs: torch.Tensor, token_embedding: nn.Embedding,
                  value_encoder: nn.Module | None = None,
                  value_modulus: int = VALUE_MODULUS) -> torch.Tensor:
    """Encode tokens, optionally replacing numeric values with Fourier features.

    This helper is shared by Neural Engine and the dense Transformer control so
    a numeric-input experiment changes the architecture, not the information
    available at the input boundary.
    """
    embedding_inputs = inputs.clamp_max(token_embedding.num_embeddings - 1)
    tokens = token_embedding(embedding_inputs)
    if value_encoder is None:
        return tokens
    values = (inputs.float() - VALUE_TOKEN_OFFSET).clamp(0, value_modulus - 1)
    angles = values.unsqueeze(-1) * (2.0 * math.pi / value_modulus)
    features = [values.unsqueeze(-1) / (value_modulus - 1)]
    for harmonic in VALUE_HARMONICS:
        features.append(torch.sin(angles * harmonic))
        features.append(torch.cos(angles * harmonic))
    numeric_tokens = value_encoder(torch.cat(features, dim=-1))
    value_mask = inputs.ge(VALUE_TOKEN_OFFSET) & inputs.lt(VALUE_TOKEN_OFFSET + value_modulus)
    return torch.where(value_mask.unsqueeze(-1), numeric_tokens, tokens)
