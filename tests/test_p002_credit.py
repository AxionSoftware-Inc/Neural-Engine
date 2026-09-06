from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from neural_engine.circuits import MicroCircuitBank
from neural_engine.p002_credit import CounterfactualCircuitCredit
from train import load_config, make_model


class _ReplayToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.circuits = MicroCircuitBank(num_circuits=4, state_dim=4, rank=2)
        self.router_probe = nn.Parameter(torch.tensor(0.5))
        self.active_circuits = 2
        self.internal_steps = 1

    def forward(
        self,
        inputs: torch.Tensor,
        adaptive: bool = False,
        forced_selected_ids: torch.Tensor | None = None,
        forced_selected_weights: torch.Tensor | None = None,
        forced_route_gains: torch.Tensor | None = None,
    ):
        del adaptive, forced_selected_weights, forced_route_gains
        if forced_selected_ids is None:
            ids = torch.tensor([[[0, 1]]], device=inputs.device).expand(inputs.shape[0], -1, -1)
        else:
            ids = forced_selected_ids
        chosen = ids[:, 0, 0]
        score = (
            self.circuits.bias[chosen, 0]
            + 0.01 * self.circuits.down[chosen].reshape(inputs.shape[0], -1).sum(dim=1)
            + 0.01 * self.circuits.up[chosen].reshape(inputs.shape[0], -1).sum(dim=1)
            + 0.0 * self.router_probe
        )
        logits = torch.stack([score, -score], dim=-1)
        return logits, {}


def _route_stats(batch_size: int) -> dict[str, torch.Tensor]:
    selected = torch.tensor([[[0, 1]]]).expand(batch_size, -1, -1).clone()
    return {
        "selected_ids": selected,
        "selected_weights": torch.full((batch_size, 1, 2), 0.5),
        "route_gains": torch.ones(batch_size, 1),
    }


def test_auxiliary_credit_changes_only_winner_circuit_row() -> None:
    torch.manual_seed(3)
    model = _ReplayToy()
    inputs = torch.ones(8, 1, dtype=torch.long)
    targets = torch.zeros(8, dtype=torch.long)
    route_stats = _route_stats(inputs.shape[0])
    logits, _ = model(inputs, forced_selected_ids=route_stats["selected_ids"])

    credit = CounterfactualCircuitCredit(
        4, 1, 2, interval=1, candidates=2, weight=0.5, min_eligible=1, seed=9
    )
    credit.observe_on_policy(route_stats["selected_ids"])
    update = credit.prepare_update(model, inputs, targets, logits, route_stats, 1)
    assert update is not None
    assert update.circuit_id in {2, 3}
    assert model.router_probe.grad is None

    loss = F.cross_entropy(logits, targets)
    loss.backward()
    before = {
        name: getattr(model.circuits, name).grad.detach().clone()
        for name in ("down", "up", "bias")
    }
    credit.apply_update(model.circuits, update)

    for name in ("down", "up", "bias"):
        after = getattr(model.circuits, name).grad
        for circuit_id in range(4):
            if circuit_id == update.circuit_id:
                assert not torch.equal(after[circuit_id], before[name][circuit_id])
            else:
                assert torch.equal(after[circuit_id], before[name][circuit_id])
    assert model.router_probe.grad is not None
    assert float(model.router_probe.grad) == 0.0


def test_credit_target_cycles_over_step_and_slot_roles() -> None:
    credit = CounterfactualCircuitCredit(
        8, internal_steps=3, active_circuits=2, interval=4, candidates=2,
        weight=0.2, min_eligible=1, seed=1,
    )
    assert [credit._target(step) for step in (4, 8, 12, 16, 20, 24, 28)] == [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (0, 0)
    ]


def test_p002_config_is_opt_in_20m_32_bank() -> None:
    config_path = Path("configs/ne_p002_20m_32.yaml")
    config = load_config(str(config_path), smoke=False)
    assert config["num_circuits"] == 32
    assert config["active_circuits"] == 2
    assert config.get("routing_mode") == "learned"
    assert config.get("route_exploration_prob", 0.0) == 0.0
    assert config.get("routing_coverage_weight", 0.0) == 0.0
    model = make_model(config)
    report = model.parameter_report()
    assert 18_000_000 <= report["total_params"] <= 22_000_000
