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


def test_composition_batch_oversamples_deeper_tasks():
    batch = SyntheticTaskGenerator(seed=6).composition_batch(1280)
    deep_fraction = float(batch.depths.ge(2).float().mean())
    assert 0.55 < deep_fraction < 0.70


def test_composition_strength_can_be_softened():
    batch = SyntheticTaskGenerator(seed=7).composition_batch(1280, strength=0.5)
    deep_fraction = float(batch.depths.ge(2).float().mean())
    assert 0.47 < deep_fraction < 0.61
