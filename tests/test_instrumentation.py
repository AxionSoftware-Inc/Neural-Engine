import torch

from baseline.transformer import DenseTransformerBaseline
from neural_engine.instrumentation import (
    count_parameters,
    estimate_neural_engine_macs,
    estimate_transformer_macs,
    parameter_bytes,
)
from neural_engine.model import NeuralEngineV0


def test_neural_engine_estimate_includes_router_and_active_path():
    model = NeuralEngineV0(vocab_size=128, num_classes=16, seq_len=8, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=3,
                           slot_count=5, numeric_value_encoding=True)
    estimate = estimate_neural_engine_macs(model, executed_steps=1.5, value_tokens=3)
    assert parameter_bytes(model) == count_parameters(model) * 4
    assert estimate["estimated_total_parameter_bytes"] == parameter_bytes(model)
    assert estimate["estimated_active_macs_per_sample"] < estimate["estimated_full_macs_per_sample"]
    assert estimate["estimated_parameter_read_bytes_per_sample"] > 0


def test_transformer_parameter_estimate_matches_reference_model():
    config = {
        "vocab_size": 128, "num_classes": 16, "seq_len": 8, "d_model": 32,
        "nhead": 4, "num_layers": 2, "ff_dim": 64,
    }
    model = DenseTransformerBaseline(**config, dropout=0.0)
    estimate = estimate_transformer_macs(config)
    assert estimate["estimated_dense_parameter_bytes"] == count_parameters(model) * 4
    assert estimate["estimated_dense_macs_per_sample"] > 0


def test_transformer_numeric_encoder_reuses_value_representation():
    model = DenseTransformerBaseline(vocab_size=128, num_classes=16, seq_len=8,
                                     d_model=32, nhead=4, num_layers=2, ff_dim=64,
                                     numeric_value_encoding=True)
    inputs = torch.tensor([[1, 2, 32, 63, 0, 0, 0, 0]])
    logits, stats = model(inputs)
    assert logits.shape == (1, 16)
    assert stats == {}
    assert model.token_embedding.num_embeddings == 16
    assert model.value_encoder is not None
