from __future__ import annotations

import torch

from neural_engine.model import NeuralEngineV0
from neural_engine.p004_cascade_credit import CascadeCredit


def _credit() -> CascadeCredit:
    return CascadeCredit(
        internal_steps=3,
        active_circuits=2,
        candidate_pool=4,
        interval=1,
        diagnostic_interval=2,
        margin=0.01,
        temperature=0.1,
        weight=0.2,
        seed=7,
    )


def test_cascade_plan_changes_only_target_step_and_leaves_suffix_on_policy() -> None:
    credit = _credit()
    selected = torch.tensor(
        [
            [[0, 1], [2, 3], [4, 5]],
            [[1, 2], [3, 4], [5, 6]],
        ]
    )
    candidates = torch.tensor(
        [
            [[0, 1, 7, 8], [2, 3, 8, 9], [4, 5, 9, 10]],
            [[1, 2, 7, 8], [3, 4, 8, 9], [5, 6, 9, 10]],
        ]
    )
    stats = {"selected_ids": selected, "candidate_ids": candidates}
    probe = credit.build_probe(stats, 1)
    assert probe is not None
    assert probe.step == 0
    assert torch.all(probe.cascade_plan[:, 1:] == -1)
    assert torch.all(probe.cascade_plan[probe.probed_mask, 0] >= 0)
    assert torch.equal(probe.fixed_suffix_plan[:, 1:], selected[:, 1:])


def test_credit_updates_router_keys_not_query() -> None:
    torch.manual_seed(3)
    model = NeuralEngineV0(
        vocab_size=128,
        num_classes=8,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=8,
        circuit_rank=4,
        router_branch=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        internal_steps=3,
    )
    credit = _credit()
    query = torch.randn(4, 16, requires_grad=True)
    current = torch.tensor([[0, 1], [0, 1], [2, 3], [2, 3]])
    alternative = torch.tensor([[0, 4], [0, 4], [2, 5], [2, 5]])
    current_loss = torch.tensor([1.0, 0.5, 0.8, 0.4])
    alternative_loss = torch.tensor([0.6, 0.7, 0.3, 0.6])
    mask = torch.ones(4, dtype=torch.bool)
    loss, info = credit.auxiliary_loss(
        model.router,
        query,
        current,
        alternative,
        current_loss,
        alternative_loss,
        mask,
    )
    loss.backward()
    assert info["credited_examples"] == 4.0
    assert query.grad is None
    assert model.router.keys.grad is not None
    touched = model.router.keys.grad.detach().norm(dim=1).gt(0)
    # The common member of each current/alternative pair cancels out.  Only
    # the replaced rows should receive the pair-ranking gradient.
    assert not touched[0] and touched[1] and not touched[2] and touched[3]
    assert touched[4] and touched[5]
    assert not touched[6] and not touched[7]


def test_real_model_accepts_changed_step_with_natural_suffix() -> None:
    torch.manual_seed(11)
    model = NeuralEngineV0(
        vocab_size=128,
        num_classes=8,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=8,
        circuit_rank=4,
        router_branch=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        internal_steps=3,
    ).eval()
    inputs = torch.tensor(
        [
            [1, 33, 34, 0, 0, 0, 0, 0],
            [2, 35, 36, 0, 0, 0, 0, 0],
            [3, 37, 38, 0, 0, 0, 0, 0],
            [4, 39, 40, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    logits, stats = model(inputs, adaptive=False)
    credit = _credit()
    probe = credit.build_probe(stats, 1)
    assert probe is not None
    replayed, replay_stats = model(
        inputs,
        adaptive=False,
        forced_selected_ids=probe.cascade_plan,
        forced_selected_weights=stats["selected_weights"],
        forced_route_gains=stats["route_gains"],
    )
    assert replayed.shape == logits.shape
    assert replay_stats["selected_ids"].shape == stats["selected_ids"].shape
    assert torch.equal(
        replay_stats["selected_ids"][probe.probed_mask, probe.step],
        probe.alternative_ids[probe.probed_mask],
    )


def test_credit_target_step_cycles_across_recurrent_cascade() -> None:
    credit = CascadeCredit(
        internal_steps=3,
        active_circuits=2,
        candidate_pool=4,
        interval=4,
        diagnostic_interval=16,
        seed=1,
    )
    assert [credit.target_step(step) for step in (4, 8, 12, 16, 20, 24)] == [0, 1, 2, 0, 1, 2]
