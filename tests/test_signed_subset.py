import torch
from torch import nn
from torch.nn import functional as F

from benchmark_qwen_multi_layer_transplant import (
    SignedSubsetReconstructionQwenChild,
    TransferredRoutedQwenChild,
    initialize_signed_subset_reconstruction,
)


class TinyQwenMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


def test_signed_subset_fit_and_oracle_forward_are_finite():
    torch.manual_seed(2026)
    parent = TinyQwenMlp()
    inputs = torch.randn(2, 3, 4)
    target = parent.down_proj(
        F.silu(parent.gate_proj(inputs)) * parent.up_proj(inputs)
    )
    base = TransferredRoutedQwenChild(
        parent, num_experts=4, active_experts=2, temperature=1.0,
        dispatch_mode="grouped", partition_mode="contiguous",
        route_source="oracle-subset", hard_route_scale=None,
    )
    child = SignedSubsetReconstructionQwenChild(base)
    initialize_signed_subset_reconstruction(
        child, [{"input": inputs, "output": target}],
        torch.device("cpu"), torch.float32,
    )
    child.eval()
    output = child(inputs)
    assert output.shape == target.shape
    assert torch.isfinite(output).all()
    assert child.last_subset_index is not None
    assert child.coefficients.shape == (6, 4)

