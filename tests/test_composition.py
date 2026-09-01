import pytest
import torch

from data.composition import (
    COMPOSITION_PAIRS,
    CompositionalProgramGenerator,
    apply_operation,
)
from neural_engine.model import NeuralEngineV0
from train_composition import evaluate


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


def test_composition_evaluate_chunks_and_returns_cpu_metrics():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=8, d_model=32,
                           state_dim=32, num_circuits=32, circuit_rank=4,
                           router_branch=2, router_depth=2, candidate_pool=4,
                           active_circuits=2, internal_steps=2, slot_count=6,
                           numeric_value_encoding=True, adaptive_halting=True)
    generator = CompositionalProgramGenerator(seed=3, split="heldout")
    result = evaluate(model, generator, examples_per_task=65, device=torch.device("cpu"))
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["avg_executed_steps"] >= 1.0
