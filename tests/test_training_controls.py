import pytest
import torch

from neural_engine.model import NeuralEngineV0
from train import apply_routing_schedule, controlled_task_route_ids


def test_controlled_task_routes_are_fixed_and_grouped():
    task_ids = torch.tensor([0, 1, 4, 7])
    routes = controlled_task_route_ids(task_ids, internal_steps=3,
                                       active_circuits=2, num_circuits=8)

    assert routes.shape == (4, 3, 2)
    assert torch.equal(routes[:, 0], routes[:, 1])
    assert torch.equal(routes[:, 1], routes[:, 2])
    assert torch.equal(routes[0, 0], torch.tensor([0, 1]))
    assert torch.equal(routes[1, 0], torch.tensor([2, 3]))
    assert torch.equal(routes[2, 0], torch.tensor([0, 1]))
    assert torch.equal(routes[3, 0], torch.tensor([6, 7]))


def test_controlled_task_routes_reject_invalid_dimensions():
    with pytest.raises(ValueError):
        controlled_task_route_ids(torch.tensor([0]), 2, 4, 2)


def test_routing_schedule_exposes_capacity_and_depth_at_requested_step():
    model = NeuralEngineV0(
        vocab_size=32, num_classes=8, seq_len=4, d_model=16, state_dim=16,
        num_circuits=16, circuit_rank=4, router_branch=2, router_depth=3,
        candidate_pool=4, active_circuits=2, internal_steps=2,
    )
    schedule = [{"step": 3, "capacity": 8, "depth": 2}]

    apply_routing_schedule(model, schedule, step=2)
    assert model.router.routing_capacity == 16
    assert model.router.active_depth == 3

    apply_routing_schedule(model, schedule, step=3)
    assert model.router.routing_capacity == 8
    assert model.router.active_depth == 2


def test_routing_schedule_requires_capacity():
    model = NeuralEngineV0(
        vocab_size=32, num_classes=8, seq_len=4, d_model=16, state_dim=16,
        num_circuits=16, circuit_rank=4, router_branch=2, router_depth=3,
        candidate_pool=4, active_circuits=2, internal_steps=2,
    )
    with pytest.raises(ValueError):
        apply_routing_schedule(model, [{"step": 1}], step=1)


def test_circuit_delta_scale_is_positive_and_defaults_to_one():
    model = NeuralEngineV0(
        vocab_size=32, num_classes=8, seq_len=4, d_model=16, state_dim=16,
        num_circuits=16, circuit_rank=4, router_branch=2, router_depth=3,
        candidate_pool=4, active_circuits=2, internal_steps=2,
    )
    assert model.circuit_delta_scale == 1.0
    with pytest.raises(ValueError):
        NeuralEngineV0(
            vocab_size=32, num_classes=8, seq_len=4, d_model=16, state_dim=16,
            num_circuits=16, circuit_rank=4, router_branch=2, router_depth=3,
            candidate_pool=4, active_circuits=2, internal_steps=2,
            circuit_delta_scale=0.0,
        )


def test_input_reinjection_schedule_matches_internal_steps():
    model = NeuralEngineV0(
        vocab_size=32, num_classes=8, seq_len=4, d_model=16, state_dim=16,
        num_circuits=16, circuit_rank=4, router_branch=2, router_depth=3,
        candidate_pool=4, active_circuits=2, internal_steps=2,
        input_reinjection_schedule=[1.0, 0.25],
    )
    assert model.input_reinjection_schedule == (1.0, 0.25)
    with pytest.raises(ValueError):
        NeuralEngineV0(
            vocab_size=32, num_classes=8, seq_len=4, d_model=16, state_dim=16,
            num_circuits=16, circuit_rank=4, router_branch=2, router_depth=3,
            candidate_pool=4, active_circuits=2, internal_steps=2,
            input_reinjection_schedule=[1.0],
        )


def test_route_bounded_correction_gate_starts_at_identity():
    model = NeuralEngineV0(
        vocab_size=32, num_classes=8, seq_len=4, d_model=16, state_dim=16,
        num_circuits=16, circuit_rank=4, router_branch=2, router_depth=3,
        candidate_pool=4, active_circuits=2, internal_steps=2,
        correction_gate_mode="route_bounded",
    )
    assert model.correction_gate is not None
    assert torch.allclose(model.correction_gate.weight,
                          torch.zeros_like(model.correction_gate.weight))
    assert torch.allclose(model.correction_gate.bias,
                          torch.zeros_like(model.correction_gate.bias))
