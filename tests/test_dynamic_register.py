import torch

from data.dynamic_composition import DynamicCompositionGenerator
from neural_engine.dynamic_register import DynamicRegisterNeuralEngine


def test_dynamic_generator_layout_and_depth_split():
    train = DynamicCompositionGenerator(max_ops=6, train_max_ops=4, split="train", seed=3)
    heldout = DynamicCompositionGenerator(max_ops=6, train_max_ops=4, split="heldout", seed=4)
    assert train.seq_len == 14
    assert train.allowed_depths == (1, 2, 3, 4)
    assert heldout.allowed_depths == (5, 6)
    batch = train.task_balanced_batch(8)
    assert tuple(batch.inputs.shape) == (8, 14)
    assert batch.stage_targets.shape == (8, 6)
    assert batch.stage_mask.shape == (8, 6)


def test_dynamic_register_forward_has_sparse_trajectory_stats():
    model = DynamicRegisterNeuralEngine(
        max_ops=4,
        seq_len=10,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
    )
    generator = DynamicCompositionGenerator(max_ops=4, train_max_ops=4, seed=5)
    batch = generator.task_balanced_batch(6)
    logits, stats = model(batch.inputs)
    assert tuple(logits.shape) == (6, 64)
    assert tuple(stats["selected_ids"].shape) == (6, 4, 4)
    assert tuple(stats["step_logits"].shape) == (6, 4, 64)
    assert torch.equal(stats["executed_steps"], batch.stage_mask.sum(dim=1))


def test_dynamic_register_handles_heldout_depths_without_attention():
    model = DynamicRegisterNeuralEngine(
        max_ops=6,
        seq_len=14,
        d_model=24,
        state_dim=24,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
    )
    generator = DynamicCompositionGenerator(max_ops=6, train_max_ops=4, split="heldout", seed=7)
    batch = generator.task_balanced_batch(4)
    logits, stats = model(batch.inputs)
    assert logits.shape[0] == 4
    assert stats["executed_steps"].min().item() >= 5
