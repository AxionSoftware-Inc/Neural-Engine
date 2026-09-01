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
