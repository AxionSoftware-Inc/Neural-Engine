import torch
from torch import nn

from benchmark_qwen_teacher_basis_swiglu import (
    TeacherBasisSwiGLU,
    activation_basis,
    initialize_teacher_basis,
)


class DummyQwenMLP(nn.Module):
    def __init__(self, hidden_size: int = 8, inner_size: int = 12) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.down_proj = nn.Linear(inner_size, hidden_size, bias=False)


def test_teacher_basis_is_orthonormal_and_initializes_child() -> None:
    torch.manual_seed(7)
    parent = DummyQwenMLP()
    inputs = torch.randn(3, 5, 8)
    basis = activation_basis(parent, inputs, 4)
    assert basis.shape == (12, 4)
    assert torch.allclose(
        basis.transpose(0, 1) @ basis,
        torch.eye(4),
        atol=1e-5,
    )

    child = TeacherBasisSwiGLU(8, 4)
    stats = initialize_teacher_basis(child, parent, inputs, basis=basis)
    assert stats["mode"] == "teacher-derived-activation-covariance"
    assert stats["basis_orthogonality_error"] < 1e-5
    output = child(inputs)
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
