import pytest

from data.composition import (
    COMPOSITION_PAIRS,
    CompositionalProgramGenerator,
    apply_operation,
)


def test_composition_generator_holds_out_only_requested_pairs():
    train = CompositionalProgramGenerator(seed=1, split="train")
    heldout = CompositionalProgramGenerator(seed=1, split="heldout")
    assert len(train.allowed_specs) == len(COMPOSITION_PAIRS) - 2
    assert len(heldout.allowed_specs) == 2
    assert {spec.name for spec in train.allowed_specs}.isdisjoint(
        {spec.name for spec in heldout.allowed_specs})


def test_composition_batch_contains_partial_targets_and_expected_tokens():
    generator = CompositionalProgramGenerator(seed=2, split="heldout")
    batch = generator.balanced_batch(examples_per_task=3)
    assert batch.inputs.shape == (6, 8)
    assert batch.stage_targets.shape == (6, 3)
    assert batch.stage_mask.all()
    assert batch.inputs[:, 0].eq(1).all()
    assert batch.inputs[:, 1:3].ge(2).logical_and(batch.inputs[:, 1:3].le(4)).all()
    assert batch.inputs[:, 3:6].ge(32).all()
    assert batch.depths.eq(3).all()


def test_apply_operation_rejects_unknown_programs():
    with pytest.raises(ValueError):
        apply_operation("divide", 4, 2)
