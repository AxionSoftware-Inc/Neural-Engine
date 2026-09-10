import os
from concurrent.futures import ThreadPoolExecutor

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


def test_native_fused_service_accepts_raw_numeric_value_tokens():
    model = NeuralEngineV0(
        vocab_size=128, num_classes=16, seq_len=8, d_model=16, state_dim=16,
        num_circuits=8, circuit_rank=2, router_branch=2, router_depth=1,
        candidate_pool=4, active_circuits=2, internal_steps=1,
        numeric_value_encoding=True, circuit_dispatch_backend="native_cuda_fused",
    ).eval()
    service = NativeFusedService(model, NativeFusedShapeCache(model, capture_graphs=False))
    result = service.infer([[1, 32, 95]])
    assert result["logit_shape"] == [1, 16]


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


def test_native_fused_service_batches_same_sequence_and_splits_results():
    base = _service()
    service = NativeFusedService(
        base.model,
        NativeFusedShapeCache(base.model, capture_graphs=False),
        max_batch_size=4,
        batch_window_ms=10.0,
    )
    rows = [[[1, 2, 3]], [[4, 5, 6]]]
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda item: service.infer(item, True), rows))
        assert [result["batch_size"] for result in results] == [1, 1]
        assert all(result["logit_shape"] == [1, 16] for result in results)
        assert all(len(result["logits"]) == 1 for result in results)
        stats = service.health()["batching"]
        assert stats["enabled"] is True
        assert stats["batch_count"] == 1
        assert stats["coalesced_request_count"] == 2
        assert stats["max_observed_batch_rows"] == 2
    finally:
        service.close()


def test_native_fused_service_explicit_batch_buckets_sequence_shapes():
    service = _service()
    requests = [
        {"inputs": [[1, 2, 3]], "return_logits": True},
        {"inputs": [[4, 5, 6]], "return_logits": True},
        {"inputs": [[7, 8]], "return_logits": True},
    ]
    result = service.infer_batch(requests, max_batch_size=2)
    assert result["request_count"] == 3
    assert result["group_count"] == 2
    assert result["groups"] == [
        {"sequence_length": 3, "batch_size": 2, "request_count": 2},
        {"sequence_length": 2, "batch_size": 1, "request_count": 1},
    ]
    assert [item["sequence_length"] for item in result["responses"]] == [3, 3, 2]
    assert all(item["logit_shape"] == [1, 16] for item in result["responses"])
    for request, response in zip(requests, result["responses"]):
        direct = service.infer(request["inputs"], return_logits=True)
        assert torch.allclose(
            torch.tensor(response["logits"]),
            torch.tensor(direct["logits"]),
            atol=1e-5,
            rtol=1e-5,
        )


def test_native_fused_service_explicit_batch_honors_configured_limit():
    service = _service()
    with pytest.raises(ValueError, match="configured limit"):
        service.infer_batch([{"inputs": [[1, 2, 3]]}], max_batch_size=9)
