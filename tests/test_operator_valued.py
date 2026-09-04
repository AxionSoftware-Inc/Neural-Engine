import torch

from neural_engine.operator_valued import OperatorValuedLinear


def test_operator_valued_forward_matches_materialized_weight():
    torch.manual_seed(17)
    layer = OperatorValuedLinear(32, 48, packet_width=16, basis_count=8)
    inputs = torch.randn(5, 3, 32)
    expected = torch.nn.functional.linear(inputs, layer.effective_weight(), layer.bias)
    actual = layer(inputs)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_standard_operator_basis_reproduces_arbitrary_block_matrix():
    torch.manual_seed(18)
    width = 4
    layer = OperatorValuedLinear(8, 8, packet_width=width, basis_count=width * width, bias=False)
    standard = torch.eye(width * width).reshape(width * width, width, width)
    target_blocks = torch.randn(2, 2, width, width)
    with torch.no_grad():
        layer.basis.copy_(standard)
        layer.coeff.copy_(target_blocks.reshape(2, 2, width * width))
    assert torch.allclose(layer.effective_blocks(), target_blocks, atol=1e-6, rtol=1e-6)


def test_operator_valued_scalar_accounting_is_explicit():
    layer = OperatorValuedLinear(384, 384, packet_width=16, basis_count=8)
    report = layer.parameter_report()
    assert report["scalar_dof"] == 8 * 16 * 16 + 24 * 24 * 8 + 384
    assert report["effective_matrix_entries"] == 384 * 384
    assert report["theoretical_macs"] == 24 * 8 * 16 * 16 + 24 * 24 * 8 * 16
