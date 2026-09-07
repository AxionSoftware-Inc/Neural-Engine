import torch
from torch import nn

from benchmark_qwen_multi_layer_transplant import (
    CrossGroupOutputMixRoutedQwenChild,
    TransferredRoutedQwenChild,
)


class TinyQwenMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)


def test_packed_dispatch_matches_token_loop() -> None:
    torch.manual_seed(2026)
    child = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "packed", "contiguous",
        "router", 2.0,
    ).eval()
    inputs = torch.randn(3, 5, 8)
    child.dispatch_mode = "token-loop"
    token_loop = child(inputs)
    child.dispatch_mode = "packed"
    packed = child(inputs)
    assert torch.allclose(token_loop, packed, atol=1e-6, rtol=1e-6)
    assert child.last_selected_outputs is not None
    assert child.last_selected_outputs.shape == (3, 5, 2, 8)


def test_packed_fused_dispatch_matches_token_loop() -> None:
    torch.manual_seed(2027)
    child = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "packed-fused", "contiguous",
        "router", 2.0,
    ).eval()
    inputs = torch.randn(3, 5, 8)
    child.dispatch_mode = "token-loop"
    token_loop = child(inputs)
    child.dispatch_mode = "packed-fused"
    packed_fused = child(inputs)
    assert torch.allclose(token_loop, packed_fused, atol=1e-6, rtol=1e-6)
    assert child.last_selected_outputs is not None
    assert child.last_selected_outputs.shape == (3, 5, 2, 8)


def test_grouped_fused_dispatch_matches_grouped() -> None:
    torch.manual_seed(2028)
    child = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "grouped-fused", "contiguous",
        "router", 2.0,
    ).eval()
    inputs = torch.randn(3, 5, 8)
    child.dispatch_mode = "grouped"
    grouped = child(inputs)
    child.dispatch_mode = "grouped-fused"
    grouped_fused = child(inputs)
    assert torch.allclose(grouped, grouped_fused, atol=1e-6, rtol=1e-6)
    assert child.last_selected_outputs is not None
    assert child.last_selected_outputs.shape == (3, 5, 2, 8)


def test_cross_group_hard_correction_matches_reference_formula() -> None:
    torch.manual_seed(2029)
    base = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "grouped", "contiguous",
        "router", 2.0,
    )
    child = CrossGroupOutputMixRoutedQwenChild(base, 3).eval()
    inputs = torch.randn(3, 5, 8)
    actual = child(inputs)
    selected_outputs = base.last_selected_outputs
    selected = base.last_selected
    route_weights = base.last_route_weights
    assert selected_outputs is not None
    assert selected is not None
    assert route_weights is not None
    base_output = base(inputs)
    latent = torch.einsum(
        "...kh,...krh->...kr", selected_outputs, child.mix_in[selected],
    )
    selected_corrections = torch.einsum(
        "...kr,...khr->...kh", latent, child.mix_out[selected],
    )
    expected = base_output + base.hard_route_scale * (
        selected_corrections * route_weights.unsqueeze(-1)
    ).sum(dim=-2)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
