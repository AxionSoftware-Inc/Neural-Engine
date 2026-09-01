import torch

from neural_engine.router import HierarchicalRouter


def test_router_returns_local_structured_selection():
    router = HierarchicalRouter(32, num_circuits=128, branch=4, depth=3, candidate_pool=16, active_circuits=4)
    selected, weights, stats = router(torch.randn(7, 32))
    assert selected.shape == (7, 4)
    assert weights.shape == (7, 4)
    assert torch.allclose(weights.sum(-1), torch.ones(7), atol=1e-5)
    assert stats["candidate_ids"].shape == (7, 16)
    assert int(selected.max()) < 128


def test_multi_address_router_keeps_total_candidate_budget_structured():
    router = HierarchicalRouter(32, num_circuits=128, branch=4, depth=3,
                                candidate_pool=16, active_circuits=4, num_addresses=2)
    selected, weights, stats = router(torch.randn(7, 32))
    assert selected.shape == (7, 4)
    assert weights.shape == (7, 4)
    assert stats["candidate_ids"].shape == (7, 16)
    assert int(stats["router_decisions"]) == 6


def test_soft_coverage_regularizer_is_differentiable():
    router = HierarchicalRouter(32, num_circuits=64, branch=4, depth=2,
                                candidate_pool=8, active_circuits=4)
    _, _, stats = router(torch.randn(7, 32), coverage=True)
    assert stats["routing_coverage_loss"].ndim == 0
    assert torch.isfinite(stats["routing_coverage_loss"])
    stats["routing_coverage_loss"].backward()
    assert router.level_projections.grad is not None


def test_router_capacity_warmup_limits_reachable_bank_and_can_expand():
    router = HierarchicalRouter(32, num_circuits=128, branch=4, depth=3,
                                candidate_pool=16, active_circuits=4,
                                routing_capacity=64, routing_depth=2)
    state = torch.randn(7, 32)
    _, _, warmup = router(state)
    assert int(warmup["router_decisions"]) == 2
    assert int(warmup["candidate_ids"].max()) < 64

    router.set_routing_state(capacity=128, depth=3)
    _, _, expanded = router(state)
    assert int(expanded["router_decisions"]) == 3
    assert int(expanded["candidate_ids"].max()) < 128


def test_router_supports_per_example_bank_windows():
    router = HierarchicalRouter(32, num_circuits=128, branch=4, depth=3,
                                candidate_pool=16, active_circuits=4)
    offsets = torch.tensor([0, 32, 64, 96, 0, 32, 64], dtype=torch.long)
    selected, _, stats = router(torch.randn(7, 32), routing_offset=offsets,
                                routing_capacity=32)
    assert (selected >= offsets.unsqueeze(-1)).all()
    assert (selected < (offsets + 32).unsqueeze(-1)).all()
    assert (stats["candidate_ids"] >= offsets.unsqueeze(-1)).all()


def test_router_supports_multiple_per_example_candidate_windows():
    router = HierarchicalRouter(32, num_circuits=128, branch=4, depth=3,
                                candidate_pool=16, active_circuits=4)
    windows = torch.tensor([
        [0, 64], [8, 72], [16, 80], [24, 88], [32, 96], [40, 96], [48, 96]
    ], dtype=torch.long)
    selected, _, stats = router(torch.randn(7, 32), routing_capacity=32,
                                routing_windows=windows)
    candidates = stats["candidate_ids"].view(7, 2, 8)
    for index in range(7):
        in_first = (selected[index].unsqueeze(-1) >= windows[index, 0]) & (
            selected[index].unsqueeze(-1) < windows[index, 0] + 32)
        in_second = (selected[index].unsqueeze(-1) >= windows[index, 1]) & (
            selected[index].unsqueeze(-1) < windows[index, 1] + 32)
        assert (in_first | in_second).all()
        assert torch.equal(torch.sort(candidates[index].reshape(-1)).values,
                           torch.sort(stats["candidate_ids"][index]).values)
