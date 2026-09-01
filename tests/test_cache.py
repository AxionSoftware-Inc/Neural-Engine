import torch

from neural_engine.cache import CircuitRowCache
from neural_engine.circuits import MicroCircuitBank


def test_cpu_circuit_cache_matches_direct_bank_and_tracks_hits():
    torch.manual_seed(3)
    bank = MicroCircuitBank(num_circuits=8, state_dim=6, rank=2)
    state = torch.randn(4, 6)
    ids = torch.tensor([[1, 2], [2, 3], [1, 3], [4, 5]])
    weights = torch.softmax(torch.randn(4, 2), dim=-1)
    expected = bank(state, ids, weights)
    cache = CircuitRowCache(bank, capacity=8, device="cpu")
    bank.set_cache(cache)
    actual = bank(state, ids, weights)
    bank(state, ids, weights)
    assert torch.allclose(actual, expected)
    assert cache.miss_rows == 5
    assert cache.hit_rows == 5
    assert cache.hit_rate == 0.5
    assert cache.h2d_bytes == 0


def test_small_cache_keeps_current_batch_rows_available_after_eviction():
    torch.manual_seed(4)
    bank = MicroCircuitBank(num_circuits=8, state_dim=4, rank=2)
    cache = CircuitRowCache(bank, capacity=1, device="cpu")
    ids = torch.tensor([[0, 1, 2]])
    weights = torch.ones(1, 3) / 3.0
    state = torch.randn(1, 4)
    cached = bank
    cached.set_cache(cache)
    cached(state, ids, weights)
    assert cache.resident_rows == 1
