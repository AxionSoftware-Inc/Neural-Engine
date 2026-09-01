from __future__ import annotations

import torch
from torch import nn

from neural_engine.encoding import VALUE_HARMONICS, encode_tokens


class DenseTransformerBaseline(nn.Module):
    """Dense control model using the same fixed synthetic input encoding."""

    def __init__(self, vocab_size: int = 128, num_classes: int = 64, seq_len: int = 32,
                 d_model: int = 384, nhead: int = 8, num_layers: int = 8,
                 ff_dim: int = 2048, dropout: float = 0.0,
                 numeric_value_encoding: bool = False):
        super().__init__()
        self.numeric_value_encoding = numeric_value_encoding
        embedding_vocab = 16 if numeric_value_encoding else vocab_size
        self.token_embedding = nn.Embedding(embedding_vocab, d_model, padding_idx=0)
        self.value_encoder = (nn.Linear(1 + 2 * len(VALUE_HARMONICS), d_model)
                              if numeric_value_encoding else None)
        self.position_embedding = nn.Parameter(torch.zeros(seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
                                           dropout=dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.output = nn.Linear(d_model, num_classes)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = encode_tokens(inputs, self.token_embedding, self.value_encoder)
        x = x + self.position_embedding[: inputs.shape[1]]
        padding = inputs.eq(0)
        x = self.encoder(x, src_key_padding_mask=padding)
        mask = (~padding).unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.output(pooled), {}
