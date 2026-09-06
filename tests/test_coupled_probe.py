import copy
from argparse import Namespace

import pytest
import torch

from benchmark_probe_router import (acceptance, body_digest, common_oracle, new_model,
                                    oracle_metrics, parameter_metrics, rollout,
                                    subset_stats, train_router)
from data.generator import SyntheticTaskGenerator
from neural_engine.coupled_probe import CoupledProbeRouter, signed_probe_loss
from neural_engine.model import NeuralEngineV0
from neural_engine.router import ProbeRouteRouter
from train import BatchSource


def small_model():
    model = NeuralEngineV0(d_model=16, state_dim=16, num_circuits=32,
                           circuit_rank=4, router_branch=4, router_depth=3,
                           candidate_pool=8, active_circuits=2, internal_steps=3)
    model.router = CoupledProbeRouter(16, 32, router_dim=8)
    return model.eval()


def test_compact_size_and_interface():
    torch.manual_seed(17)
    router = CoupledProbeRouter(128, 32)
    old = ProbeRouteRouter(128, 32)
    assert sum(p.numel() for p in router.parameters()) < sum(p.numel() for p in old.parameters()) / 4
    state = torch.randn(9, 128)
    selected, weights, stats = router(state, coverage=True)
    assert selected.shape == (9, 2)
    assert stats["candidate_ids"].shape == (9, 8)
    assert stats["candidate_pair_scores"].shape == (9, 28)
    assert selected[:, 0].ne(selected[:, 1]).all()
    assert (selected[:, :, None] == stats["candidate_ids"][:, None, :]).any(-1).all()
    assert torch.equal(weights, torch.full_like(weights, 0.5))
    assert torch.equal(stats["route_gain"], torch.ones(9))
    assert not bool(stats["soft_route"])
    stats["routing_coverage_loss"].backward()
    assert router.keys.grad is not None
    assert router.retriever_query.weight.grad is not None


def test_pair_scores_symmetric_and_consistent():
    router = CoupledProbeRouter(16, 32)
    state = torch.randn(5, 16)
    _, _, stats = router(state)
    for index, positions in enumerate(router.pair_positions):
        pairs = stats["candidate_ids"][:, positions]
        assert torch.allclose(router.selection_scores(state, pairs),
                              stats["candidate_pair_scores"][:, index], atol=1e-6)
        assert torch.allclose(router.selection_scores(state, pairs.flip(-1)),
                              router.selection_scores(state, pairs), atol=1e-6)


def test_validation_capacity_exploration_and_serialization():
    router = CoupledProbeRouter(16, 32)
    state = torch.randn(100, 16)
    router.set_routing_state(capacity=16)
    selected, _, stats = router(state, exploration_prob=1, coverage=True)
    assert int(selected.max()) < 16
    assert selected[:, 0].ne(selected[:, 1]).all()
    assert stats["exploration_mask"].all()
    assert torch.isfinite(stats["routing_coverage_loss"])
    with pytest.raises(ValueError):
        router(state, routing_capacity=7)
    with pytest.raises(ValueError):
        router(state, target_bases=torch.zeros(100, dtype=torch.long))
    with pytest.raises(ValueError):
        router.selection_scores(state, torch.full((100, 2), -1))
    with pytest.raises(ValueError):
        router.selection_scores(state, torch.zeros(100, 2, dtype=torch.long))
    router.eval()
    assert torch.equal(router(state, exploration_prob=1)[0], router(state)[0])
    cloned = copy.deepcopy(router)
    cloned.load_state_dict(router.state_dict())
    assert torch.equal(cloned(state)[0], router(state)[0])


def test_probe_gradient_reaches_excluded_key_but_not_body_query():
    torch.manual_seed(1)
    router = CoupledProbeRouter(16, 32)
    query = torch.randn(3, 16, requires_grad=True)
    selected, _, stats = router(query)
    outsiders = []
    for pool in stats["candidate_ids"]:
        outsiders.append(next(i for i in range(32) if i not in pool.tolist()))
    outsiders = torch.tensor(outsiders)
    alternative = torch.stack((selected[:, 0], outsiders), -1)
    delta = torch.tensor([0.4, -0.3, 0.2], requires_grad=True)
    loss, metrics = signed_probe_loss(router, query, selected, alternative, delta)
    loss.backward()
    assert query.grad is None and delta.grad is None
    assert router.keys.grad[outsiders].abs().sum() > 0
    assert router.interaction_query.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(value) for value in metrics.values())


def test_empty_probe_is_finite_differentiable_zero():
    router = CoupledProbeRouter(16, 32)
    loss, _ = signed_probe_loss(router, torch.empty(0, 16), torch.empty(0, 2, dtype=torch.long),
                                 torch.empty(0, 2, dtype=torch.long), torch.empty(0))
    assert loss.item() == 0
    loss.backward()


def test_model_api_forced_replay_and_exactly_two_body_rows():
    torch.manual_seed(3)
    model = small_model()
    inputs = torch.randint(1, 64, (4, 8))
    observed = []
    handle = model.circuits.register_forward_pre_hook(lambda module, args: observed.append(args[1].clone()))
    before = body_digest(model)
    logits, stats = model(inputs, adaptive=False, coverage=True)
    assert len(observed) == 3
    assert all(ids.shape == (4, 2) for ids in observed)
    assert stats["candidate_ids"].shape == (4, 3, 8)
    assert torch.isfinite(stats["routing_coverage_loss"])
    replay, _ = model(inputs, adaptive=False, forced_selected_ids=stats["selected_ids"],
                       forced_selected_weights=stats["selected_weights"],
                       forced_route_gains=stats["route_gains"])
    assert torch.allclose(logits, replay, atol=1e-6)
    sentinel, _ = model(inputs, adaptive=False, forced_selected_ids=torch.full_like(stats["selected_ids"], -1))
    assert torch.allclose(logits, sentinel, atol=1e-6)
    alternative = stats["selected_ids"][:, 1].flip(-1)
    replay_one = rollout(model, inputs, subset_stats(stats, slice(None)), 1, alternative)
    assert torch.allclose(logits, replay_one, atol=1e-6)
    handle.remove()
    assert before == body_digest(model)
    measured = parameter_metrics(model)
    assert measured["active_circuit_parameters_per_decision"] == 2 * (16 * 4 * 2 + 16)
    assert measured["active_parameters_per_decision_upper_bound"] < measured["total_parameters"]


def test_common_oracle_exact_cost_and_regret_decomposition():
    torch.manual_seed(7)
    model = small_model()
    batch = BatchSource(SyntheticTaskGenerator(seed=19), 16, torch.device("cpu")).balanced(1)
    # One example is sufficient for the exhaustive evaluation-only unit test.
    batch.inputs, batch.targets = batch.inputs[:1], batch.targets[:1]
    pairs, costs, queries = common_oracle(model, batch, chunk_size=64)
    assert pairs.shape == (496, 2)
    assert costs.shape == (1, 3, 496)
    _, stats = model(batch.inputs, adaptive=False)
    direct = rollout(model, batch.inputs, stats, 1, pairs[17:18])
    direct_ce = torch.nn.functional.cross_entropy(direct, batch.targets)
    assert torch.allclose(costs[0, 1, 17], direct_ce, atol=1e-5)
    metrics = oracle_metrics(model.router, pairs, costs, queries)
    assert 0 <= metrics["candidate_recall"] <= 1
    assert metrics["selection_regret_ce"] >= -1e-6
    assert metrics["retrieval_regret_ce"] >= -1e-6
    assert metrics["mean_regret_ce"] == pytest.approx(
        metrics["selection_regret_ce"] + metrics["retrieval_regret_ce"], abs=1e-6)


def test_training_smoke_preserves_frozen_body_for_both_objectives():
    torch.manual_seed(17)
    teacher = NeuralEngineV0(d_model=16, state_dim=16, num_circuits=32,
                             circuit_rank=4, candidate_pool=8, active_circuits=2,
                             internal_steps=3).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    config = {"seed": 17, "seq_len": 32}
    args = Namespace(steps=4, imitation_steps=1, learning_rate=3e-4,
                     examples_per_task=1, log_every=0)
    for variant in ("old", "old-signed", "coupled"):
        model = new_model(teacher, 17, variant)
        initial = body_digest(model)
        result = train_router(teacher, model, config, variant, args)
        assert result["body_unchanged"] and body_digest(model) == initial
        assert sum(result["probe_counts"].values()) > 0
        assert all(p.grad is None for name, p in model.named_parameters()
                   if not name.startswith("router."))


def test_ce_only_improvement_is_rejected():
    old = {"hard_accuracy": 0.5, "ce": 2.5, "p95_regret_ce": 1.0,
           "candidate_recall": 0.2, "latency_batch": {"median_ms": 2.0},
           "dead_circuits": 0}
    new = dict(old, ce=2.3, p95_regret_ce=0.8)
    rows = [{"seed": seed, "models": {"old": old, "coupled": new,
                                        "hierarchical": old, "old-signed": old}}
            for seed in (17, 18)]
    assert acceptance(rows)["decision"] == "reject_for_adoption"
    new["hard_accuracy"] = 0.53
    assert acceptance(rows)["decision"] == "screen_pass_not_scaling_proof"
