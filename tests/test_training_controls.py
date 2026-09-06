import pytest
import torch

from train import controlled_task_route_ids


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
