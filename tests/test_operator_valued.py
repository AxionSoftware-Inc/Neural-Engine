import torch

from neural_engine.operator_valued import (
    OperationConditionedOperatorValuedLinear,
    OperatorValuedLinear,
)


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


def test_operation_conditioned_operator_matches_per_operation_materialization():
    torch.manual_seed(19)
    layer = OperationConditionedOperatorValuedLinear(
        32, 48, num_operations=3, packet_width=16, basis_count=8
    )
    inputs = torch.randn(5, 3, 32)
    operation_ids = torch.tensor([
        [0, 1, 2], [2, 1, 0], [1, 0, 2], [0, 2, 1], [2, 0, 1]
    ])
    actual = layer(inputs, operation_ids)
    expected = torch.stack([
        torch.nn.functional.linear(
            inputs[row, col],
            layer.effective_weight(int(operation_ids[row, col])),
            layer.bias[int(operation_ids[row, col])],
        )
        for row in range(inputs.shape[0])
        for col in range(inputs.shape[1])
    ]).reshape_as(actual)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_operation_conditioned_operator_reports_shared_basis_and_per_op_dof():
    layer = OperationConditionedOperatorValuedLinear(
        384, 384, num_operations=3, packet_width=16, basis_count=8
    )
    report = layer.parameter_report()
    assert report["scalar_dof"] == 8 * 16 * 16 + 3 * 24 * 24 * 8 + 3 * 384
    assert report["effective_matrix_entries"] == 3 * 384 * 384
    assert report["theoretical_macs"] == 24 * 8 * 16 * 16 + 3 * 24 * 24 * 8 * 16
