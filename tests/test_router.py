import torch

from neural_engine.router import (FactorizedRouter, FlatRouter, HierarchicalRouter,
                                  ProbeRouteRouter)


def test_router_returns_local_structured_selection():
    router = HierarchicalRouter(32, num_circuits=128, branch=4, depth=3, candidate_pool=16, active_circuits=4)
    selected, weights, stats = router(torch.randn(7, 32))
    assert selected.shape == (7, 4)
    assert weights.shape == (7, 4)
    assert torch.allclose(weights.sum(-1), torch.ones(7), atol=1e-5)
    assert stats["candidate_ids"].shape == (7, 16)
    assert int(selected.max()) < 128


def test_soft_training_route_uses_candidate_pool_but_eval_returns_topk():
    router = HierarchicalRouter(32, num_circuits=128, branch=4, depth=3,
                                candidate_pool=16, active_circuits=4,
                                soft_routing_temperature=0.5)
    router.train()
    selected, weights, stats = router(torch.randn(7, 32))
    assert selected.shape == (7, 16)
    assert weights.shape == (7, 16)
    assert torch.allclose(weights.sum(-1), torch.ones(7), atol=1e-5)
    assert bool(stats["soft_route"])

    router.eval()
    selected, weights, stats = router(torch.randn(7, 32))
    assert selected.shape == (7, 4)
    assert weights.shape == (7, 4)
    assert not bool(stats["soft_route"])


def test_router_target_supervision_reaches_tree_and_keys():
    router = HierarchicalRouter(32, num_circuits=32, branch=4, depth=3,
                                candidate_pool=8, active_circuits=2)
    state = torch.randn(7, 32)
    _, _, stats = router(state, target_bases=torch.tensor([0, 2, 4, 6, 8, 10, 12]))
    assert torch.isfinite(stats["routing_target_loss"])
    stats["routing_target_loss"].backward()
    assert router.level_projections.grad is not None
    assert router.keys.grad is not None


def test_flat_router_scores_full_bank_but_executes_topk():
    router = FlatRouter(32, num_circuits=32, candidate_pool=8, active_circuits=2)
    state = torch.randn(7, 32)
    selected, weights, stats = router(state)
    assert selected.shape == (7, 2)
    assert weights.shape == (7, 2)
    assert stats["candidate_ids"].shape == (7, 8)
    assert int(stats["candidate_ids"].max()) < 32
    loss = selected.float().mean() * 0.0 + weights.square().mean()
    loss.backward()
    assert router.keys.grad is not None


def test_probe_router_retrieves_pool_and_scores_pairs():
    router = ProbeRouteRouter(32, num_circuits=32, candidate_pool=8,
                              active_circuits=2, pair_rank=4)
    state = torch.randn(7, 32)
    selected, weights, stats = router(state)
    assert selected.shape == (7, 2)
    assert weights.shape == (7, 2)
    assert stats["candidate_ids"].shape == (7, 8)
    assert stats["candidate_pair_scores"].shape == (7, 28)
    assert torch.allclose(weights, torch.full_like(weights, 0.5))
    loss = router.candidate_pair_scores(state, stats["candidate_ids"]).mean()
    loss.backward()
    assert router.utility_keys.grad is not None
    assert router.pair_embeddings.grad is not None
    router.zero_grad(set_to_none=True)
    router.retriever_scores(state).mean().backward()
    assert router.retriever_query.weight.grad is not None


def test_factorized_router_exploration_uses_uniform_weights_for_sampled_ids():
    router = FactorizedRouter(16, num_circuits=16, factor_count=4,
                              factor_candidate_pool=2, candidate_pool=4,
                              active_circuits=2)
    router.train()
    torch.manual_seed(7)
    selected, weights, _ = router(torch.randn(64, 16), exploration_prob=1.0)
    assert selected.shape == (64, 2)
    assert torch.allclose(weights, torch.full_like(weights, 0.5))


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


def test_task_reuse_regularizer_is_differentiable():
    router = HierarchicalRouter(32, num_circuits=64, branch=4, depth=2,
                                candidate_pool=8, active_circuits=4)
    _, _, stats = router(torch.randn(8, 32), reuse_task_ids=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))
    assert stats["routing_reuse_loss"].ndim == 0
    assert torch.isfinite(stats["routing_reuse_loss"])
    stats["routing_reuse_loss"].backward()
    assert router.level_projections.grad is not None


def test_task_reuse_regularizer_handles_uneven_groups():
    router = HierarchicalRouter(16, num_circuits=32, branch=2, depth=3,
                                candidate_pool=4, active_circuits=2)
    _, _, stats = router(torch.randn(7, 16), reuse_task_ids=torch.tensor([0, 0, 0, 1, 1, 2, 2]))
    assert torch.isfinite(stats["routing_reuse_loss"])


def test_task_reuse_can_skip_coarse_levels():
    router = HierarchicalRouter(16, num_circuits=32, branch=2, depth=3,
                                candidate_pool=4, active_circuits=2)
    _, _, stats = router(
        torch.randn(8, 16),
        reuse_task_ids=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        reuse_start_level=2,
    )
    assert torch.isfinite(stats["routing_reuse_loss"])


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
