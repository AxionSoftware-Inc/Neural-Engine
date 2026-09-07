import torch
from torch import nn

from benchmark_qwen_adaptive_width_oracle import OracleAdaptiveWidthChild, parse_widths


class AddConstant(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.value


def test_adaptive_wrapper_executes_selected_width_only() -> None:
    inputs = torch.zeros(2, 3, 4)
    route = torch.tensor([[0, 1, 2], [2, 1, 0]])
    child = OracleAdaptiveWidthChild(
        [AddConstant(1.0), AddConstant(2.0), AddConstant(3.0)], route,
    )
    output = child(inputs)
    expected = torch.tensor([
        [[1.0] * 4, [2.0] * 4, [3.0] * 4],
        [[3.0] * 4, [2.0] * 4, [1.0] * 4],
    ])
    assert torch.equal(output, expected)


def test_width_parser_requires_sorted_unique_positive_widths() -> None:
    assert parse_widths("192,768,1536") == [192, 768, 1536]
    for invalid in ("0,4", "4,4", "8,4"):
        try:
            parse_widths(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {invalid}")
