import torch
from torch import nn

from benchmark_qwen_multi_layer_transplant import TransferredRoutedQwenChild


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
