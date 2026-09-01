import torch

from neural_engine.optim import LazyAdamW


def test_lazy_adamw_updates_only_rows_with_gradients():
    parameter = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    optimizer = LazyAdamW([parameter], lr=0.1, betas=(0.0, 0.0),
                          weight_decay=0.0, lazy_parameters=[parameter])
    before = parameter.detach().clone()
    parameter.grad = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    optimizer.step()
    assert not torch.equal(parameter[0], before[0])
    assert torch.equal(parameter[1], before[1])
    report = optimizer.report()
    assert report["lazy_rows_last_step"] == 1
    assert report["lazy_state_rows"] == 1


def test_lazy_adamw_applies_decay_only_to_touched_rows():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = LazyAdamW([parameter], lr=0.1, betas=(0.0, 0.0),
                          weight_decay=0.1, lazy_parameters=[parameter])
    parameter.grad = torch.zeros_like(parameter)
    parameter.grad[0, 0] = 1.0
    optimizer.step()
    assert torch.all(parameter[0] < 1.0)
    assert torch.equal(parameter[1], torch.ones(2))


def test_lazy_adamw_accepts_batched_active_row_ids():
    parameter = torch.nn.Parameter(torch.ones(4, 3))
    optimizer = LazyAdamW([parameter], lr=0.1, betas=(0.0, 0.0),
                          weight_decay=0.0, lazy_parameters=[parameter])
    parameter.grad = torch.zeros_like(parameter)
    parameter.grad[1] = 1.0
    parameter.grad[3] = 1.0
    optimizer.set_active_rows({parameter: torch.tensor([1, 3])})
    optimizer.step()
    assert torch.equal(parameter[0], torch.ones(3))
    assert torch.all(parameter[1] < 1.0)
    assert torch.equal(parameter[2], torch.ones(3))
    assert torch.all(parameter[3] < 1.0)
    assert optimizer.report()["lazy_state_rows"] == 2
