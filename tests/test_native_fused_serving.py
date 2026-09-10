import pytest
import torch
from concurrent.futures import ThreadPoolExecutor

from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0
from neural_engine.native_fused_serving import NativeFusedShapeCache


def _model(**overrides):
    config = dict(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=16, circuit_rank=4, router_branch=2, router_depth=2,
        candidate_pool=4, active_circuits=2, internal_steps=1,
        circuit_bank_mode="factorized", factor_count=4,
        ordered_factor_slots=True, circuit_dispatch_backend="native_cuda_fused",
        factor_address_layout="stable_prefix", legacy_factor_count=2,
    )
    config.update(overrides)
    return NeuralEngineV0(**config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
def test_native_fused_shape_cache_reuses_graph_and_preserves_logits():
    model = _model().cuda().eval()
    inputs = SyntheticTaskGenerator(seq_len=8, seed=601).task_balanced_batch(1, "cuda").inputs
    cache = NativeFusedShapeCache(model, max_shapes=2, warmup_iters=2)
    with torch.inference_mode():
        eager, _ = model(inputs, adaptive=False, collect_stats=False)
        first = cache(inputs)
        second = cache(inputs.clone())
    assert torch.allclose(first, eager, atol=1e-5, rtol=1e-5)
    assert torch.equal(first, second)
    stats = cache.stats()
    assert stats["capture_count"] == 1
    assert stats["cache_hit_count"] == 1
    assert stats["eager_fallback_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
def test_native_fused_shape_cache_evicts_old_shape_and_dynamic_width_falls_back():
    model = _model().cuda().eval()
    generator = SyntheticTaskGenerator(seq_len=8, seed=602)
    first = generator.task_balanced_batch(1, "cuda").inputs
    second = generator.task_balanced_batch(2, "cuda").inputs
    cache = NativeFusedShapeCache(model, max_shapes=1, warmup_iters=2)
    cache(first)
    cache(second)
    cache(first)
    assert cache.stats()["eviction_count"] == 2

    dynamic = _model(dynamic_width_mode="learned", dynamic_width_min=1,
                     dynamic_width_threshold=0.5).cuda().eval()
    dynamic_cache = NativeFusedShapeCache(dynamic, warmup_iters=2)
    dynamic_cache(first)
    dynamic_stats = dynamic_cache.stats()
    assert dynamic_stats["capture_count"] == 0
    assert dynamic_stats["eager_fallback_count"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
def test_native_fused_shape_cache_capture_failure_uses_eager_fallback(monkeypatch):
    model = _model().cuda().eval()
    inputs = SyntheticTaskGenerator(seq_len=8, seed=603).task_balanced_batch(1, "cuda").inputs
    cache = NativeFusedShapeCache(model, warmup_iters=2)

    def fail_capture(*_args, **_kwargs):
        raise RuntimeError("synthetic graph capture failure")

    monkeypatch.setattr(torch.cuda, "make_graphed_callables", fail_capture)
    output = cache(inputs)
    stats = cache.stats()
    assert output.shape == (1, 64)
    assert stats["capture_count"] == 0
    assert stats["eager_fallback_count"] == 1
    assert stats["capture_failures"][0]["type"] == "RuntimeError"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
def test_native_fused_shape_cache_serializes_same_stream_callers():
    model = _model().cuda().eval()
    inputs = SyntheticTaskGenerator(seq_len=8, seed=604).task_balanced_batch(1, "cuda").inputs
    cache = NativeFusedShapeCache(model, warmup_iters=2)
    reference = cache(inputs)

    def request():
        return cache(inputs.clone())

    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(lambda _index: request(), range(4)))
    torch.cuda.synchronize()
    assert all(torch.allclose(output, reference, atol=1e-5, rtol=1e-5)
               for output in outputs)
    assert cache.stats()["capture_count"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
def test_native_fused_shape_cache_serializes_different_shapes_on_same_stream():
    model = _model().cuda().eval()
    generator = SyntheticTaskGenerator(seq_len=8, seed=606)
    first = generator.task_balanced_batch(1, "cuda").inputs
    second = generator.task_balanced_batch(2, "cuda").inputs
    cache = NativeFusedShapeCache(model, warmup_iters=2)
    with torch.inference_mode():
        first_reference, _ = model(first, adaptive=False, collect_stats=False)
        second_reference, _ = model(second, adaptive=False, collect_stats=False)

    def request(index):
        return cache((first if index % 2 == 0 else second).clone())

    with ThreadPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(request, range(8)))
    torch.cuda.synchronize()
    assert all(torch.allclose(outputs[index], first_reference, atol=1e-5, rtol=1e-5)
               for index in (0, 2, 4, 6))
    assert all(torch.allclose(outputs[index], second_reference, atol=1e-5, rtol=1e-5)
               for index in (1, 3, 5, 7))
    assert cache.stats()["capture_count"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph requires CUDA")
def test_native_fused_shape_cache_rejects_cross_process_reuse(monkeypatch):
    model = _model().cuda().eval()
    inputs = SyntheticTaskGenerator(seq_len=8, seed=605).task_balanced_batch(1, "cuda").inputs
    cache = NativeFusedShapeCache(model, warmup_iters=2)
    monkeypatch.setattr("neural_engine.native_fused_serving.os.getpid",
                        lambda: cache.owner_pid + 1)
    with pytest.raises(RuntimeError, match="process-local"):
        cache(inputs)
