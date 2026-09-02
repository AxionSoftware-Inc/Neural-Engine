import pytest
import torch
from torch import nn

from data.composition import (
    COMPOSITION_PAIRS,
    CompositionalProgramGenerator,
    apply_operation,
)
from neural_engine.model import NeuralEngineV0
from neural_engine.register_model import TypedRegisterNeuralEngine
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


def test_typed_register_model_exposes_serial_register_graph_and_gradients():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, route_exploration_prob=0.1,
    )
    generator = CompositionalProgramGenerator(seed=4, split="heldout")
    batch = generator.task_balanced_batch(8)
    logits, stats = model(batch.inputs)
    assert logits.shape == (8, 64)
    assert stats["selected_ids"].shape == (8, 3, 2)
    assert stats["step_logits"].shape == (8, 3, 64)
    assert stats["register_norms"].shape == (8, 3)
    loss = nn.functional.cross_entropy(logits, batch.targets)
    loss.backward()
    assert model.circuits.down.grad is not None
    assert model.operation_embedding.weight.grad is not None


def test_typed_register_direct_readout_skips_third_bank_lookup():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, readout_mode="direct",
    )
    generator = CompositionalProgramGenerator(seed=5, split="heldout")
    batch = generator.task_balanced_batch(4)
    logits, stats = model(batch.inputs)
    assert logits.shape == (4, 64)
    assert stats["selected_ids"][:, 2].eq(-1).all()
    assert stats["executed_steps"].eq(3).all()


def test_typed_register_partitions_routes_by_operator_and_role():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, typed_route_partitions=True,
        operator_partition_count=4,
    )
    generator = CompositionalProgramGenerator(seed=6, split="heldout")
    batch = generator.task_balanced_batch(8)
    _, stats = model(batch.inputs)
    partition_size = 8
    op_ids = (batch.inputs[:, 1:3] - 2)
    first = stats["selected_ids"][:, 0]
    second = stats["selected_ids"][:, 1]
    third = stats["selected_ids"][:, 2]
    assert (first >= op_ids[:, 0].unsqueeze(-1) * partition_size).all()
    assert (first < (op_ids[:, 0] + 1).unsqueeze(-1) * partition_size).all()
    assert (second >= op_ids[:, 1].unsqueeze(-1) * partition_size).all()
    assert (second < (op_ids[:, 1] + 1).unsqueeze(-1) * partition_size).all()
    assert (third >= 3 * partition_size).all()
    assert (third < 4 * partition_size).all()


def test_typed_register_shared_private_routes_allow_shared_or_private_bank():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, typed_route_partitions=True,
        typed_route_shared=True, operator_partition_count=4,
    )
    generator = CompositionalProgramGenerator(seed=7, split="heldout")
    batch = generator.task_balanced_batch(8)
    _, stats = model(batch.inputs)
    partition_size = 8
    op_ids = batch.inputs[:, 1:3] - 2
    first = stats["selected_ids"][:, 0]
    second = stats["selected_ids"][:, 1]
    third = stats["selected_ids"][:, 2]
    for selected, op in ((first, op_ids[:, 0]), (second, op_ids[:, 1])):
        private_start = (op + 1).unsqueeze(-1) * partition_size
        private_end = private_start + partition_size
        shared = selected < partition_size
        private = (selected >= private_start) & (selected < private_end)
        assert (shared | private).all()
    assert (third < partition_size).all()


def test_typed_register_typed_route_query_reuses_route_for_same_operator():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, route_query_mode="typed",
    )
    model.eval()
    first = torch.tensor([[1, 2, 4, 32, 33, 34, 0, 0],
                         [1, 2, 4, 35, 36, 37, 0, 0]])
    _, first_stats = model(first)
    _, second_stats = model(first.flip(0))
    assert torch.equal(first_stats["selected_ids"], second_stats["selected_ids"].flip(0))


def test_typed_register_compressed_route_query_keeps_value_context_path():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, route_query_mode="compressed",
        route_context_dim=4,
    )
    generator = CompositionalProgramGenerator(seed=8, split="heldout")
    batch = generator.task_balanced_batch(4)
    logits, stats = model(batch.inputs)
    assert logits.shape == (4, 64)
    assert torch.isfinite(stats["router_entropy"])
    assert model.route_value_encoder[1].out_features == 4


def test_typed_register_family_local_router_keeps_shared_fallback_and_family_locality():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=2, router_depth=3,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, routing_mode="family_local",
        family_count=9, shared_fraction=0.125,
    )
    generator = CompositionalProgramGenerator(seed=9, split="heldout")
    batch = generator.task_balanced_batch(8)
    logits, stats = model(batch.inputs)
    assert logits.shape == (8, 64)
    assert stats["selected_ids"].shape == (8, 3, 2)
    assert stats["family_ids"].shape == (8, 3)
    assert model.router.shared_count == 8
    assert 0.0 <= float(stats["shared_selected_fraction"]) <= 1.0


def test_typed_register_family_local_router_rejects_legacy_partition_flags():
    with pytest.raises(ValueError):
        TypedRegisterNeuralEngine(
            vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
            num_circuits=64, circuit_rank=4, router_branch=2, router_depth=3,
            candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
            routing_mode="family_local", typed_route_partitions=True,
        )


def test_typed_register_role_anchored_router_separates_role_and_value_queries():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=2, router_depth=3,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, routing_mode="role_anchored",
        anchor_branch=2, anchor_depth=2,
    )
    generator = CompositionalProgramGenerator(seed=10, split="heldout")
    batch = generator.task_balanced_batch(8)
    logits, stats = model(batch.inputs)
    assert logits.shape == (8, 64)
    assert stats["selected_ids"].shape == (8, 3, 2)
    assert stats["anchor_ids"].shape == (8, 3)
    assert model.router.anchor_count == 4
    assert model.parameter_report()["routing_mode"] == "role_anchored"


def test_typed_register_fixed_role_cell_router_keeps_full_local_candidate_pool():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=2, router_depth=3,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, routing_mode="role_cell", role_count=9,
    )
    generator = CompositionalProgramGenerator(seed=11, split="heldout")
    batch = generator.task_balanced_batch(8)
    logits, stats = model(batch.inputs)
    assert logits.shape == (8, 64)
    assert stats["selected_ids"].shape == (8, 3, 2)
    assert stats["role_cell_ids"].shape == (8, 3)
    assert model.router.candidate_pool == 4
    assert model.parameter_report()["routing_mode"] == "role_cell"


def test_typed_register_shared_residual_bank_preserves_sparse_route_shape():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=2, router_depth=3,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, circuit_bank_mode="shared_residual",
        shared_rank=2,
    )
    generator = CompositionalProgramGenerator(seed=12, split="heldout")
    batch = generator.task_balanced_batch(8)
    logits, stats = model(batch.inputs)
    assert logits.shape == (8, 64)
    assert stats["selected_ids"].shape == (8, 3, 2)
    assert model.circuits.shared_down.shape == (32, 2)
    assert model.parameter_report()["circuit_bank_mode"] == "shared_residual"


def test_typed_register_multiplicative_pair_mode_adds_structured_interaction():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=2, router_depth=3,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, pair_mode="multiplicative",
    )
    generator = CompositionalProgramGenerator(seed=13, split="heldout")
    batch = generator.task_balanced_batch(8)
    logits, _ = model(batch.inputs)
    assert logits.shape == (8, 64)
    assert model.pair_product_encoder is not None
    assert model.parameter_report()["pair_mode"] == "multiplicative"


def test_typed_register_factorized_bank_preserves_virtual_capacity_and_route_shape():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=2, router_depth=3,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, pair_mode="multiplicative",
        circuit_bank_mode="factorized", factor_count=8,
    )
    generator = CompositionalProgramGenerator(seed=14, split="heldout")
    batch = generator.task_balanced_batch(8)
    logits, stats = model(batch.inputs)
    assert logits.shape == (8, 64)
    assert stats["selected_ids"].shape == (8, 3, 2)
    assert model.circuits.factor_count == 8
    assert model.parameter_report()["circuit_bank_mode"] == "factorized"


def test_typed_register_factorized_router_addresses_factor_pairs():
    model = TypedRegisterNeuralEngine(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=2, router_depth=3,
        candidate_pool=4, active_circuits=2, internal_steps=3, slot_count=6,
        numeric_value_encoding=True, pair_mode="multiplicative",
        routing_mode="factorized", circuit_bank_mode="factorized",
        factor_count=8,
    )
    generator = CompositionalProgramGenerator(seed=15, split="heldout")
    batch = generator.task_balanced_batch(8)
    logits, stats = model(batch.inputs)
    assert logits.shape == (8, 64)
    assert stats["selected_ids"].shape == (8, 3, 2)
    assert model.router.factor_candidate_pool == 2
    assert model.parameter_report()["routing_mode"] == "factorized"
