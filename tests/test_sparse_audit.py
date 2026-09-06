from argparse import Namespace

import pytest
import torch

from benchmark_sparse_credit_audit import (calibration_drift, calibration_refit,
                                          cascade_diagnostic, replay_args, train_credit)
from data.generator import SyntheticTaskGenerator
from neural_engine.instrumentation import estimate_neural_engine_macs
from neural_engine.model import NeuralEngineV0
from neural_engine.sparse_audit import (CircuitLedger, final_margin_loss,
                                        router_accounting, sparse_cost_report,
                                        sparse_row_credit, training_phase_budget)
from train import BatchSource


def small():
    torch.manual_seed(17)
    return NeuralEngineV0(d_model=16, state_dim=16, num_circuits=32, circuit_rank=4,
                           candidate_pool=8, active_circuits=2, internal_steps=3,
                           numeric_value_encoding=True, seq_len=8, slot_count=5).eval()


def batch():
    return BatchSource(SyntheticTaskGenerator(seq_len=8, seed=19), 16, torch.device("cpu")).balanced(1)


def test_cost_counts_frozen_parameters_all_value_positions_and_active_depth():
    model = small()
    b = batch()
    with torch.no_grad():
        _, stats = model(b.inputs, adaptive=False)
    total = sum(p.numel() for p in model.parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    cost = sparse_cost_report(model, b.inputs, stats)
    assert cost["total_parameters"] == total > 0
    assert cost["total_parameter_bytes"] == total * 4
    assert cost["mac_components"]["value_encoder"] == 8 * 13 * 16
    assert cost["mac_components"]["recurrent_gru_per_step"] == 6 * 16 * 16
    assert cost["mac_components"]["circuit_body_per_step"] == 2 * 2 * 16 * 4
    assert all(f == 2 * m for f, m in zip(cost["inference_arithmetic_flops_per_example"], cost["inference_macs_per_example"]))
    model.router.set_routing_state(depth=2)
    r = router_accounting(model.router, 16)
    assert r["projection_macs"] == 2 * 16 * 8
    assert max(cost["unique_active_parameters_per_example"]) <= total


def test_parameter_union_not_multiplied_by_recurrent_steps():
    model = small()
    b = batch()
    _, stats = model(b.inputs, adaptive=False)
    repeated = {k: v.detach().clone() for k, v in stats.items()}
    repeated["selected_ids"][:] = torch.tensor([0, 1])
    expanded = {k: v.clone() for k, v in repeated.items()}
    expanded["selected_ids"][:] = torch.tensor([[0, 1], [2, 3], [4, 5]])
    a = sparse_cost_report(model, b.inputs, repeated)
    c = sparse_cost_report(model, b.inputs, expanded)
    row = 2 * 16 * 4 + 16
    assert all(y - x == 4 * row for x, y in zip(a["unique_active_parameters_per_example"], c["unique_active_parameters_per_example"]))
    assert a["inference_macs_per_example"] == c["inference_macs_per_example"]


def test_legacy_json_keys_preserved_by_optional_v2_block():
    model = small()
    b = batch()
    _, stats = model(b.inputs, adaptive=False)
    legacy = estimate_neural_engine_macs(model, 3, value_tokens=4)
    merged = {**legacy, "sparse_audit_v2": sparse_cost_report(model, b.inputs, stats)}
    assert {key: merged[key] for key in legacy} == legacy


def test_ledger_separates_usage_probe_forwards_gradients_and_momentum_updates():
    model = small()
    ledger = CircuitLedger(model)
    q = torch.randn(2, 16)
    with ledger.phase("main", torch.tensor([0, 1])):
        model.circuits(q, torch.tensor([[0, 1], [0, 2]]), torch.full((2, 2), 0.5))
    with ledger.phase("training_probe", torch.tensor([0, 1])):
        model.circuits(q, torch.tensor([[0, 3], [0, 4]]), torch.full((2, 2), 0.5))
    optimizer = torch.optim.AdamW(model.circuits.parameters(), lr=0.01, weight_decay=0)
    for row in (0, 1):
        optimizer.zero_grad(set_to_none=True)
        for p in model.circuits.parameters():
            p.grad = torch.zeros_like(p)
            p.grad[row] = 1
        ledger.before_step()
        optimizer.step()
        ledger.after_step()
    report = ledger.report()
    ledger.close()
    assert report["usage"][0] == 2
    assert report["usage"][3] == 0
    assert report["forward_count_by_phase"]["training_probe"][3] == 1
    assert report["nonzero_gradient_steps"][0] == 1
    assert report["actual_update_count"][0] == 2
    assert report["task_usage_counts"][0][:2] == [1, 1]


def test_final_margin_is_not_cross_entropy_and_has_hard_boundary():
    logits = torch.tensor([[3., 1., 0.], [0., 2., 1.]], requires_grad=True)
    targets = torch.tensor([0, 0])
    loss = final_margin_loss(logits, targets)
    assert torch.equal(loss, torch.tensor([0., 3.]))
    loss.sum().backward()
    assert torch.equal(logits.grad[0], torch.zeros(3))


def test_sparse_credit_does_not_accumulate_router_or_controller_gradients():
    model = small()
    b = batch()
    with torch.no_grad():
        _, stats = model(b.inputs, adaptive=False)
    alternative = stats["selected_ids"][:, 1].clone()
    replacement = (alternative[:, 0] + 1) % 32
    alternative[:, 1] = replacement
    plan, weights, gains = replay_args(stats, slice(None), 1, alternative)
    loss, updates = sparse_row_credit(model, b.inputs, b.targets, plan, weights, gains, replacement)
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.parameters())
    for p, rows, gradient in updates:
        assert set(rows.tolist()) == set(replacement.tolist())
        p.grad = torch.zeros_like(p)
        p.grad.index_add_(0, rows, gradient)
        untouched = torch.ones(32, dtype=torch.bool)
        untouched[rows] = False
        assert p.grad[untouched].eq(0).all()


def test_cascade_prefix_parity_and_zero_drift_for_identical_models():
    model = small()
    b = batch()
    diagnostic = cascade_diagnostic(model, b)
    assert all(row["prefix_query_max_error"] < 1e-6 for row in diagnostic["steps"])
    assert diagnostic["steps"][-1]["suffix_rerouting_ce_effect"] == pytest.approx(0, abs=1e-6)
    drift = calibration_drift(model, model, b)
    assert drift["mean_absolute_target_drift"] == 0


def test_control_and_credit_training_preserve_non_bank_parameters():
    base = small()
    config = {"seed": 17, "seq_len": 8}
    args = Namespace(steps=2, train_examples_per_task=1, lr=1e-4, credit_weight=0.1,
                     log_every=0, calibration_steps=2)
    for arm in ("control", "underused_margin", "underused_ce", "uniform_margin"):
        trained, report = train_credit(base, config, arm, args)
        assert report["non_bank_weights_unchanged"]
        assert report["optimizer_steps_observed"] == 2
        assert sum(report["forward_count_by_phase"]["main"]) == 2 * 15 * 3 * 2
        assert sum(report["forward_count_by_phase"]["training_probe"]) == 2 * 3 * 3 * 2
    for mode in ("static", "on_policy"):
        trained = calibration_refit(base, config, mode, args)
        assert all(torch.equal(p, dict(base.named_parameters())[n])
                   for n, p in trained.named_parameters() if n != "router.keys")


def test_training_budget_never_merges_probe_cost_with_inference():
    ledger = {"forward_count_by_phase": {"main": [6, 6], "training_probe": [3, 3]},
              "optimizer_steps_observed": 1}
    cost = {"inference_macs_per_example": [100, 100]}
    result = training_phase_budget(ledger, cost)
    assert result["phases"]["main"]["forward_macs"] == 200
    assert result["phases"]["training_probe"]["forward_macs"] == 100
    assert result["backward_macs"] is None
    assert cost == {"inference_macs_per_example": [100, 100]}
