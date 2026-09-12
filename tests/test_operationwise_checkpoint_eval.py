import torch

from benchmark_operationwise_checkpoints import fixed_operation_batch
from data.composition import apply_operation


def _config():
    return {
        "max_ops": 4,
        "train_max_ops": 2,
        "seq_len": 10,
        "target_offset": 134217728,
    }


def test_fixed_operation_batch_uses_homogeneous_operation_and_exact_targets():
    batch = fixed_operation_batch(
        _config(), "multiply", depth=3, count=4, seed=7, device=torch.device("cpu")
    )
    assert tuple(batch.inputs.shape) == (4, 10)
    assert torch.equal(batch.inputs[:, 1], torch.full((4,), 4))
    assert torch.equal(batch.inputs[:, 2], torch.full((4,), 4))
    assert torch.equal(batch.inputs[:, 3], torch.full((4,), 4))
    assert torch.equal(batch.inputs[:, 4], torch.zeros(4, dtype=torch.long))
    for row in range(4):
        values = [int(value) - 32 for value in batch.inputs[row, 5:9]]
        expected = values[0]
        for value in values[1:]:
            expected = apply_operation("multiply", expected, value, modulus=None)
        assert int(batch.targets[row]) == expected + 134217728


def test_fixed_operation_batch_marks_only_executed_stages():
    batch = fixed_operation_batch(
        _config(), "subtract", depth=3, count=2, seed=11, device=torch.device("cpu")
    )
    assert torch.equal(batch.stage_mask, torch.tensor([
        [True, True, True, False],
        [True, True, True, False],
    ]))
    assert torch.equal(batch.inputs[:, 1:4], torch.full((2, 3), 3))
