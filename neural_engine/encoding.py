from __future__ import annotations

import math

import torch
from torch import nn


VALUE_TOKEN_OFFSET = 32
VALUE_MODULUS = 64
VALUE_HARMONICS = (1, 2, 4, 8, 16, 32)


def encode_tokens(inputs: torch.Tensor, token_embedding: nn.Embedding,
                  value_encoder: nn.Module | None = None) -> torch.Tensor:
    """Encode tokens, optionally replacing numeric values with Fourier features.

    This helper is shared by Neural Engine and the dense Transformer control so
    a numeric-input experiment changes the architecture, not the information
    available at the input boundary.
    """
    embedding_inputs = inputs.clamp_max(token_embedding.num_embeddings - 1)
    tokens = token_embedding(embedding_inputs)
    if value_encoder is None:
        return tokens
    values = (inputs.float() - VALUE_TOKEN_OFFSET).clamp(0, VALUE_MODULUS - 1)
    angles = values.unsqueeze(-1) * (2.0 * math.pi / VALUE_MODULUS)
    features = [values.unsqueeze(-1) / (VALUE_MODULUS - 1)]
    for harmonic in VALUE_HARMONICS:
        features.append(torch.sin(angles * harmonic))
        features.append(torch.cos(angles * harmonic))
    numeric_tokens = value_encoder(torch.cat(features, dim=-1))
    value_mask = inputs.ge(VALUE_TOKEN_OFFSET) & inputs.lt(VALUE_TOKEN_OFFSET + VALUE_MODULUS)
    return torch.where(value_mask.unsqueeze(-1), numeric_tokens, tokens)
