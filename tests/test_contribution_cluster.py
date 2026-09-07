import torch
from torch import nn

from benchmark_qwen_multi_layer_transplant import (
    contribution_cluster_partition,
    contribution_diverse_partition,
)


class TinyQwenMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)


def test_contribution_cluster_is_balanced_and_deterministic():
    torch.manual_seed(2026)
    parent = TinyQwenMlp()
    io_batches = [{"input": torch.randn(2, 4, 8)}]
    first = contribution_cluster_partition(
        parent, io_batches, torch.device("cpu"), torch.float32, 4,
        max_tokens=8, sketch_dim=4,
    )
    second = contribution_cluster_partition(
        parent, io_batches, torch.device("cpu"), torch.float32, 4,
        max_tokens=8, sketch_dim=4,
    )
    assert [int(item.numel()) for item in first] == [4, 4, 4, 4]
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert torch.cat(first).unique().numel() == 16


def test_contribution_diverse_spreads_each_cluster_across_groups():
    torch.manual_seed(2026)
    parent = TinyQwenMlp()
    io_batches = [{"input": torch.randn(2, 4, 8)}]
    clusters = contribution_cluster_partition(
        parent, io_batches, torch.device("cpu"), torch.float32, 4,
    )
    groups = contribution_diverse_partition(
        parent, io_batches, torch.device("cpu"), torch.float32, 4,
    )
    assert [int(item.numel()) for item in groups] == [4, 4, 4, 4]
    assert torch.cat(groups).unique().numel() == 16
    for cluster in clusters:
        for group in groups:
            assert torch.isin(group, cluster).sum().item() == 1
