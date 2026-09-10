from __future__ import annotations

import torch

from benchmark_route_capacity_clamp import evaluate_mode
from neural_engine.model import NeuralEngineV0


def make_small_model() -> NeuralEngineV0:
    return NeuralEngineV0(
        d_model=32, state_dim=32, num_circuits=64, circuit_rank=4,
        router_branch=4, router_depth=3, candidate_pool=8, active_circuits=2,
        internal_steps=2,
    ).eval()


def test_capacity_clamp_only_changes_reachable_router_state() -> None:
    model = make_small_model()
    batch = type("Batch", (), {
        "inputs": torch.randint(0, 128, (4, 8)),
        "targets": torch.randint(0, 64, (4,)),
    })()
    result = evaluate_mode(model, batch, "prefix_1408_depth4")
    assert result["routing_capacity"] == 64
    assert result["routing_depth"] == 3
    assert 0.0 <= result["accuracy"] <= 1.0
    assert model.router.routing_capacity == 64
    assert model.router.active_depth == 3


def test_natural_mode_preserves_configured_route_state() -> None:
    model = make_small_model()
    model.router.set_routing_state(capacity=32, depth=2)
    batch = type("Batch", (), {
        "inputs": torch.randint(0, 128, (4, 8)),
        "targets": torch.randint(0, 64, (4,)),
    })()
    result = evaluate_mode(model, batch, "natural")
    assert result["routing_capacity"] == 32
    assert result["routing_depth"] == 2
