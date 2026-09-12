import torch

from benchmark_numeric_state_distillation import digit_distillation_loss


def test_digit_distillation_is_zero_for_identical_logits():
    logits = (
        torch.randn(3, 4, 5),
        torch.randn(3, 4, 7),
    )
    mask = torch.tensor([
        [True, True, False, False],
        [True, False, False, False],
        [True, True, True, True],
    ])
    loss = digit_distillation_loss(
        {"digit_logits": logits},
        {"digit_logits": logits},
        mask,
        temperature=2.0,
    )
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_digit_distillation_ignores_unexecuted_steps():
    student = (
        torch.zeros(2, 3, 4),
        torch.zeros(2, 3, 6),
    )
    teacher = (
        torch.zeros(2, 3, 4),
        torch.zeros(2, 3, 6),
    )
    mask = torch.tensor([[True, False, False], [True, True, False]])
    baseline = digit_distillation_loss(
        {"digit_logits": student}, {"digit_logits": teacher}, mask, 2.0
    )
    changed_teacher = (
        torch.tensor([[[0.0, 0.0, 0.0, 0.0], [90.0, -90.0, 0.0, 0.0], [90.0, -90.0, 0.0, 0.0]]]),
        torch.tensor([[[0.0] * 6, [90.0, -90.0, 0.0, 0.0, 0.0, 0.0], [90.0, -90.0, 0.0, 0.0, 0.0, 0.0]]]),
    )
    changed_teacher = tuple(value.expand(2, -1, -1) for value in changed_teacher)
    changed = digit_distillation_loss(
        {"digit_logits": student}, {"digit_logits": changed_teacher}, mask, 2.0
    )
    assert torch.allclose(baseline, torch.zeros_like(baseline), atol=1e-6)
    assert changed.item() > 0.0
