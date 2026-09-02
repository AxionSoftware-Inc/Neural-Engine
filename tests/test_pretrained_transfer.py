import torch
from torch import nn

from neural_engine.pretrained_transfer import SwiGLUCircuitBank, top_contribution_circuits


class TinyQwenMlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(hidden_states))
            * self.up_proj(hidden_states)
        )


def test_qwen_style_swiglu_conversion_is_exact_with_padding_and_biases():
    torch.manual_seed(41)
    source = TinyQwenMlp(hidden_size=13, intermediate_size=37, bias=True)
    bank = SwiGLUCircuitBank.from_qwen_mlp(source, chunk_size=8)
    inputs = torch.randn(5, 7, 13)

    expected = source(inputs)
    actual = bank(inputs)

    assert bank.num_circuits == 5
    assert bank.chunk_sizes.tolist() == [8, 8, 8, 8, 5]
    assert torch.max(torch.abs(expected - actual)).item() < 2e-6
    assert torch.allclose(bank.forward_selected(
        inputs,
        torch.arange(bank.num_circuits).expand(5, 7, -1),
    ), expected, atol=2e-6, rtol=2e-6)


def test_qwen_style_swiglu_conversion_preserves_no_bias_parameter_layout():
    torch.manual_seed(42)
    source = TinyQwenMlp(hidden_size=16, intermediate_size=32, bias=False)
    bank = SwiGLUCircuitBank.from_qwen_mlp(source, chunk_size=16)
    inputs = torch.randn(11, 16)

    assert torch.allclose(bank(inputs), source(inputs), atol=2e-6, rtol=2e-6)
    assert bank.parameter_report()["total_parameters"] == 3 * 16 * 32
    assert not any(name.endswith("bias") for name, _ in bank.named_parameters())


def test_selected_circuits_are_a_controlled_sparse_approximation():
    torch.manual_seed(43)
    source = TinyQwenMlp(hidden_size=12, intermediate_size=48, bias=False)
    bank = SwiGLUCircuitBank.from_qwen_mlp(source, chunk_size=8)
    inputs = torch.randn(32, 12)
    all_ids = torch.arange(bank.num_circuits).expand(inputs.shape[0], -1)
    selected = top_contribution_circuits(bank, inputs, active_circuits=3)

    exact = bank(inputs)
    sparse = bank.forward_selected(inputs, selected)
    assert selected.shape == (32, 3)
    assert torch.allclose(bank.forward_selected(inputs, all_ids), exact, atol=2e-6, rtol=2e-6)
    assert torch.isfinite(sparse).all()
    assert not torch.allclose(sparse, exact)

