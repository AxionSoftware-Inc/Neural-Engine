import os

import pytest
import torch

from neural_engine.model import NeuralEngineV0
from neural_engine.native_fused_server import NativeFusedService
from neural_engine.native_fused_serving import NativeFusedShapeCache


def _service():
    model = NeuralEngineV0(
        vocab_size=32, num_classes=16, seq_len=8, d_model=16, state_dim=16,
        num_circuits=8, circuit_rank=2, router_branch=2, router_depth=1,
        candidate_pool=4, active_circuits=2, internal_steps=1,
        circuit_bank_mode="factorized", factor_count=4,
        ordered_factor_slots=True, circuit_dispatch_backend="native_cuda_fused",
    ).eval()
    return NativeFusedService(model, NativeFusedShapeCache(model, capture_graphs=False))


def test_native_fused_service_returns_predictions_and_optional_logits():
    service = _service()
    result = service.infer([[1, 2, 3], [4, 5, 6]], return_logits=True)
    assert result["batch_size"] == 2
    assert result["sequence_length"] == 3
    assert result["logit_shape"] == [2, 16]
    assert len(result["predictions"]) == 2
    assert len(result["logits"]) == 2
    assert result["cache"]["eager_fallback_count"] == 1


@pytest.mark.parametrize(
    "rows, message",
    [
        ([], "non-empty"),
        ([[1], [2, 3]], "same sequence"),
        ([[True]], "integers"),
        ([[-1]], "token ids must be in"),
        ([[1] * 9], "exceeds model limit"),
    ],
)
def test_native_fused_service_validates_input(rows, message):
    with pytest.raises(ValueError, match=message):
        _service().infer(rows)


def test_native_fused_service_rejects_cross_process_use(monkeypatch):
    service = _service()
    monkeypatch.setattr("neural_engine.native_fused_server.os.getpid", lambda: service.owner_pid + 1)
    with pytest.raises(RuntimeError, match="process-local"):
        service.health()
