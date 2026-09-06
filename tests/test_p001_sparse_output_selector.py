from __future__ import annotations

import torch

from neural_engine.model import NeuralEngineV0
from neural_engine.p001_sparse_output_selector import (
    ExistingKeyPairSelector,
    SparseOutputSignatureSelector,
    SparseStepContext,
    candidate_local_teacher,
    selector_cost_report,
    sparse_selector_rollout,
)


def _model() -> NeuralEngineV0:
    torch.manual_seed(17)
    model = NeuralEngineV0(
        vocab_size=128,
        num_classes=64,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=8,
        circuit_rank=4,
        router_branch=2,
        router_depth=3,
        candidate_pool=4,
        active_circuits=2,
        internal_steps=3,
        slot_count=5,
        numeric_value_encoding=True,
    )
    model.eval()
    return model


def _inputs(batch: int = 5) -> torch.Tensor:
    rows = []
    for index in range(batch):
        rows.append([1 + index % 3, 32 + index, 33 + index, 34 + index, 35 + index, 0, 0, 0])
    return torch.tensor(rows, dtype=torch.long)


def test_existing_key_selector_rollout_matches_native_model() -> None:
    model = _model()
    inputs = _inputs()
    native_logits, native_stats = model(inputs, adaptive=False)
    selector = ExistingKeyPairSelector(model.router.keys, model.router.candidate_pool)
    replay_logits, replay_stats, _ = sparse_selector_rollout(model, selector, inputs)
    torch.testing.assert_close(replay_logits, native_logits, rtol=1e-5, atol=1e-6)
    assert torch.equal(replay_stats["selected_ids"], native_stats["selected_ids"])
    assert torch.equal(replay_stats["candidate_ids"], native_stats["candidate_ids"])
    torch.testing.assert_close(
        replay_stats["selected_weights"], native_stats["selected_weights"], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        replay_stats["route_gains"], native_stats["route_gains"], rtol=1e-5, atol=1e-6
    )


def test_selector_gradient_touches_only_retrieved_signature_rows() -> None:
    torch.manual_seed(4)
    selector = SparseOutputSignatureSelector(
        state_dim=16,
        num_circuits=10,
        candidate_pool=4,
        signature_rank=2,
        signature_dim=6,
    )
    query = torch.randn(3, 16)
    candidates = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long
    )
    scores, _ = selector(query, candidates)
    scores.sum().backward()
    for parameter in (
        selector.signature_down,
        selector.signature_up,
        selector.signature_bias,
    ):
        grad = parameter.grad.reshape(parameter.shape[0], -1).abs().sum(dim=1)
        assert bool(grad[:4].gt(0).all())
        assert bool(grad[4:].eq(0).all())
    assert bool(selector.circuit_bias.grad[:4].abs().gt(0).all())
    assert bool(selector.circuit_bias.grad[4:].eq(0).all())


def test_candidate_teacher_ignores_out_of_pool_real_circuit_rows() -> None:
    model = _model()
    inputs = _inputs(batch=4)
    targets = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    encoded = model.encode(inputs)
    _, stats = model(inputs, adaptive=False)
    step = 0
    candidates = stats["candidate_ids"][:, step].clone()
    used = set(int(value) for value in candidates.reshape(-1).tolist())
    outside = next(index for index in range(model.router.num_circuits) if index not in used)
    query = stats["query_states"][:, step]
    context = SparseStepContext(
        step=step,
        query=query,
        state_before=query - model.step_embedding[step],
        encoded=encoded,
        candidate_ids=candidates,
        route_gain=stats["route_gains"][:, step],
    )
    before = candidate_local_teacher(model, context, targets)["full_local_losses"]
    with torch.no_grad():
        model.circuits.down[outside].fill_(100.0)
        model.circuits.up[outside].fill_(100.0)
        model.circuits.bias[outside].fill_(100.0)
    after = candidate_local_teacher(model, context, targets)["full_local_losses"]
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_inference_cost_report_forbids_dense_real_circuit_probe() -> None:
    model = _model()
    selector = SparseOutputSignatureSelector(
        state_dim=model.state_dim,
        num_circuits=model.router.num_circuits,
        candidate_pool=model.router.candidate_pool,
        signature_rank=2,
        signature_dim=8,
    )
    report = selector_cost_report(model, selector)
    assert report["selector_candidate_rows_touched_per_decision"] == 4
    assert report["real_circuit_rows_executed_per_decision"] == 2
    assert report["candidate_real_circuit_rows_scored_at_inference"] == 0
    assert report["full_bank_real_circuit_rows_scored_per_decision"] == 0
    assert report["dense_bank_circuit_outputs_per_decision"] == 0
