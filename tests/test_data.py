import torch

from data.generator import SyntheticTaskGenerator


def test_generator_shapes_and_exact_targets():
    batch = SyntheticTaskGenerator(seq_len=32, seed=3).batch(128)
    assert batch.inputs.shape == (128, 32)
    assert batch.targets.shape == (128,)
    assert int(batch.targets.min()) >= 0
    assert int(batch.targets.max()) < 64
    assert batch.inputs.dtype == torch.long
    assert batch.stage_targets.shape == (128, 3)
    assert batch.stage_mask.shape == (128, 3)
    assert batch.stage_mask.dtype == torch.bool
