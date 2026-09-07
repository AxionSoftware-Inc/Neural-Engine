import torch
from torch import nn

from benchmark_qwen_multi_layer_transplant import core_overlap_partition


class TinyQwenMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 16, bias=False)
        self.up_proj = nn.Linear(4, 16, bias=False)
        self.down_proj = nn.Linear(16, 4, bias=False)


def test_core_overlap_is_balanced_deterministic_and_repeats_only_core():
    torch.manual_seed(2026)
    parent = TinyQwenMlp()
    io_batches = [{"input": torch.randn(2, 3, 4)}]
    first = core_overlap_partition(
        parent, io_batches, torch.device("cpu"), torch.float32, 4,
    )
    second = core_overlap_partition(
        parent, io_batches, torch.device("cpu"), torch.float32, 4,
    )
    assert [int(item.numel()) for item in first] == [4, 4, 4, 4]
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    counts = torch.bincount(torch.cat(first), minlength=16)
    assert int((counts == 4).sum()) == 1
    assert int((counts == 1).sum()) == 12
    assert int((counts == 0).sum()) == 3

