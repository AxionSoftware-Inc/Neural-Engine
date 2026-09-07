import torch
from torch import nn

from benchmark_qwen_multi_layer_transplant import TransferredRoutedQwenChild


class TinyQwenMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)


def test_group_sketch_router_features_are_signed_and_sized() -> None:
    torch.manual_seed(2026)
    child = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "token-loop", "contiguous",
        "subset-router", 2.0, router_input="group-sketch", router_sketch_dim=3,
    )
    features = child._router_features(torch.randn(2, 5, 8))
    assert features.shape == (2, 5, 12)
    assert torch.isfinite(features).all()
    assert features.abs().sum() > 0
