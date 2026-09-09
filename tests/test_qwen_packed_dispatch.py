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


def test_grouped_prepacked_dispatch_matches_grouped() -> None:
    torch.manual_seed(2037)
    child = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "grouped-prepacked", "contiguous",
        "router", 2.0,
    ).eval()
    inputs = torch.randn(3, 5, 8)
    child.dispatch_mode = "grouped"
    grouped = child(inputs)
    child.dispatch_mode = "grouped-prepacked"
    prepacked = child(inputs)
    assert torch.allclose(grouped, prepacked, atol=1e-6, rtol=1e-6)
    assert child.last_selected_outputs is not None
    assert child._grouped_prepacked_weights is not None
    assert all(weight.is_contiguous() for weight in child._grouped_prepacked_weights)


def test_grouped_prepacked_fused_dispatch_matches_grouped_fused() -> None:
    torch.manual_seed(2038)
    child = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "grouped-prepacked-fused", "contiguous",
        "router", 2.0,
    ).eval()
    inputs = torch.randn(3, 5, 8)
    child.dispatch_mode = "grouped-fused"
    grouped_fused = child(inputs)
    child.dispatch_mode = "grouped-prepacked-fused"
    prepacked_fused = child(inputs)
    assert torch.allclose(grouped_fused, prepacked_fused, atol=1e-6, rtol=1e-6)


def test_grouped_cached_metadata_matches_grouped() -> None:
    torch.manual_seed(2026)
    child = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "grouped-cached", "contiguous",
        "router", 2.0,
    ).eval()
    inputs = torch.randn(3, 5, 8)
    child.dispatch_mode = "grouped"
    grouped = child(inputs)
    child.dispatch_mode = "grouped-cached"
    cached = child(inputs)
    assert torch.allclose(grouped, cached, atol=1e-6, rtol=1e-6)
    assert len(child._grouped_pair_metadata_cache) == 1


def test_grouped_correction_fusion_matches_vectorized_correction() -> None:
    torch.manual_seed(2035)
    base = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "grouped", "contiguous",
        "router", 2.0,
    )
    child = CrossGroupOutputMixRoutedQwenChild(base, 3).eval()
    inputs = torch.randn(3, 5, 8)
    child.correction_dispatch_backend = "vectorized"
    expected = child(inputs)
    child.correction_dispatch_backend = "grouped-fused-correction"
    actual = child(inputs)
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
    assert base.last_selected_outputs is None


def test_grouped_uniform_accumulation_matches_weighted_subset() -> None:
    torch.manual_seed(2036)
    base = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "grouped", "contiguous",
        "subset-router", 2.0,
    ).eval()
    inputs = torch.randn(3, 5, 8)
    base.grouped_uniform_accum = False
    expected = base(inputs)
    base.grouped_uniform_accum = True
    actual = base(inputs)
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)


def test_grouped_single_token_fast_path_matches_token_loop() -> None:
    torch.manual_seed(2030)
    child = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "grouped", "contiguous",
        "router", 2.0,
    ).eval()
    child.single_token_fast_path = True
    inputs = torch.randn(1, 1, 8)
    child.dispatch_mode = "token-loop"
    token_loop = child(inputs)
    child.dispatch_mode = "grouped"
    grouped = child(inputs)
    assert torch.allclose(token_loop, grouped, atol=1e-6, rtol=1e-6)
    assert child.last_selected_outputs is not None
    assert child.last_selected_outputs.shape == (1, 1, 2, 8)


def test_grouped_single_token_fast_path_supports_batched_decode() -> None:
    torch.manual_seed(2031)
    child = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "grouped", "contiguous",
        "router", 2.0,
    ).eval()
    child.single_token_fast_path = True
    inputs = torch.randn(2, 1, 8)
    child.dispatch_mode = "token-loop"
    token_loop = child(inputs)
    child.dispatch_mode = "grouped"
    grouped = child(inputs)
    assert torch.allclose(token_loop, grouped, atol=1e-6, rtol=1e-6)
    assert child.last_selected_outputs is not None
    assert child.last_selected_outputs.shape == (2, 1, 2, 8)


def test_cross_group_hard_correction_matches_reference_formula() -> None:
    torch.manual_seed(2029)
    base = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "grouped", "contiguous",
        "router", 2.0,
    )
    child = CrossGroupOutputMixRoutedQwenChild(base, 3).eval()
    inputs = torch.randn(3, 5, 8)
    direct = child(inputs)
    child.max_dense_gather_bytes = 0
    packed = child(inputs)
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
    assert torch.allclose(direct, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(packed, expected, atol=1e-6, rtol=1e-6)


def test_cross_group_single_token_batched_matmul_matches_reference() -> None:
    torch.manual_seed(2032)
    base = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "grouped", "contiguous",
        "router", 2.0,
    )
    child = CrossGroupOutputMixRoutedQwenChild(base, 3).eval()
    inputs = torch.randn(3, 1, 8)
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


def test_cross_group_effective_output_projection_matches_correction() -> None:
    torch.manual_seed(2034)
    base = TransferredRoutedQwenChild(
        TinyQwenMlp(), 4, 2, 1.0, "grouped", "contiguous",
        "router", 2.0,
    )
    child = CrossGroupOutputMixRoutedQwenChild(base, 3).eval()
    base.single_token_fast_path = True
    inputs = torch.randn(3, 1, 8)
    with torch.inference_mode():
        expected = child(inputs)
        child.correction_dispatch_backend = "effective-output"
        actual = child(inputs)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_single_token_bmm_projection_matches_einsum() -> None:
    torch.manual_seed(2033)
    parent = TinyQwenMlp()
    einsum_child = TransferredRoutedQwenChild(
        parent, 4, 2, 1.0, "grouped", "contiguous", "router", 2.0,
    ).eval()
    bmm_child = TransferredRoutedQwenChild(
        parent, 4, 2, 1.0, "grouped", "contiguous", "router", 2.0,
    ).eval()
    bmm_child.load_state_dict(einsum_child.state_dict())
    einsum_child.single_token_fast_path = True
    bmm_child.single_token_fast_path = True
    bmm_child.single_token_projection_backend = "bmm"
    inputs = torch.randn(3, 1, 8)
    with torch.inference_mode():
        expected = einsum_child(inputs)
        actual = bmm_child(inputs)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
