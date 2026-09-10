import torch

from data.generator import SyntheticTaskGenerator
from data.tasks import TASKS


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


def test_generator_respects_disjoint_value_range():
    batch = SyntheticTaskGenerator(seed=8, value_min=32, value_max=39).batch(128)
    operands = batch.inputs[:, 1:5]
    non_padding = operands.ne(0)
    assert int(operands[non_padding].min()) >= 32 + 32
    assert int(operands[non_padding].max()) <= 39 + 32


def test_combination_split_is_disjoint():
    train = SyntheticTaskGenerator(seed=10, split="train")
    heldout = SyntheticTaskGenerator(seed=11, split="heldout")
    for generator, expected_bucket in ((train, {0, 1, 2}), (heldout, {3})):
        for task in TASKS:
            for _ in range(32):
                tokens, _, _, _ = generator._one(task)
                values = [token - 32 for token in tokens[1:1 + task.arity]]
                assert generator._combination_bucket(task, values) in expected_bucket


def test_mixed_value_batch_includes_requested_edge_range():
    generator = SyntheticTaskGenerator(seed=12)
    batch = generator.mixed_value_batch(
        512, edge_fraction=1.0, edge_value_min=56, edge_value_max=63,
        task_balanced=True)
    operands = batch.inputs[:, 1:5]
    non_padding = operands.ne(0)
    values = operands[non_padding] - 32
    assert int(values.min()) >= 56
    assert int(values.max()) <= 63


def test_mixed_value_batch_can_use_multiple_edge_ranges():
    generator = SyntheticTaskGenerator(seed=13)
    batch = generator.mixed_value_batch(
        1024, edge_fraction=1.0, edge_ranges=[(0, 7), (56, 63)],
        task_balanced=True)
    operands = batch.inputs[:, 1:5]
    values = operands[operands.ne(0)] - 32
    assert bool(values.le(7).any())
    assert bool(values.ge(56).any())
    assert bool(values[(values > 7) & (values < 56)].numel() == 0)
