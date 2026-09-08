import torch

from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0
from neural_engine.router import StableFamilyRouter


def test_neural_engine_forward_and_gradients():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=64, state_dim=64,
                           num_circuits=64, circuit_rank=8, router_branch=4, router_depth=3,
                           candidate_pool=8, active_circuits=2, internal_steps=2)
    batch = SyntheticTaskGenerator(seed=4).batch(8)
    logits, stats = model(batch.inputs)
    loss = torch.nn.functional.cross_entropy(logits, batch.targets)
    loss.backward()
    assert logits.shape == (8, 64)
    assert model.circuits.down.grad is not None
    assert model.router.level_projections.grad is not None
    assert stats["selected_ids"].shape == (8, 2, 2)
    assert stats["route_deltas"].shape == (8, 2, 64)
    assert stats["step_logits"].shape == (8, 2, 64)


def test_input_reinjection_scale_is_configurable():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=2,
                           input_reinjection=0.5)
    batch = SyntheticTaskGenerator(seed=5).batch(3)
    logits, _ = model(batch.inputs)
    assert logits.shape == (3, 64)
    assert model.input_reinjection == 0.5


def test_inference_can_skip_diagnostic_route_stats_without_changing_logits():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=2)
    batch = SyntheticTaskGenerator(seed=51).batch(3)
    with torch.no_grad():
        with_stats, _ = model(batch.inputs, adaptive=False, collect_stats=True)
        without_stats, stats = model(batch.inputs, adaptive=False, collect_stats=False)
    assert torch.allclose(with_stats, without_stats)
    assert stats == {}


def test_fixed_nonadaptive_stats_free_path_matches_stats_path():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=2)
    batch = SyntheticTaskGenerator(seed=52).batch(3)
    with torch.no_grad():
        stats_logits, _ = model(batch.inputs, adaptive=False, collect_stats=True)
        fixed, _ = model(batch.inputs, adaptive=False, collect_stats=False)
    assert torch.allclose(stats_logits, fixed)


def test_gated_memory_write_preserves_forward_and_gradients():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=2,
                           memory_write_mode="gated")
    batch = SyntheticTaskGenerator(seed=6).batch(3)
    logits, _ = model(batch.inputs)
    torch.nn.functional.cross_entropy(logits, batch.targets).backward()
    assert logits.shape == (3, 64)
    assert model.memory_write.weight.grad is not None


def test_position_conditioning_preserves_operand_order():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=2)
    first = torch.zeros(1, 32, dtype=torch.long)
    first[0, :3] = torch.tensor([1, 32 + 7, 32 + 11])
    second = first.clone()
    second[0, 1], second[0, 2] = second[0, 2], second[0, 1]
    assert not torch.allclose(model.encode(first), model.encode(second))


def test_slot_encoder_preserves_structured_input():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=2, slot_count=5)
    batch = SyntheticTaskGenerator(seed=8).batch(3)
    encoded = model.encode(batch.inputs)
    assert encoded.shape == (3, 32)


def test_task_context_binds_operation_identity():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=2, slot_count=5,
                           task_context=True)
    first = torch.zeros(1, 32, dtype=torch.long)
    first[0, :3] = torch.tensor([1, 32 + 7, 32 + 11])
    second = first.clone()
    second[0, 0] = 2
    with torch.no_grad():
        first_encoded = model.encode(first)
        first_logits, _ = model(first)
        second_logits, _ = model(second)
    assert not torch.allclose(first_encoded, model.encode(second))
    assert not torch.allclose(first_logits, second_logits)


def test_typed_register_bridge_reinjects_predicted_value_and_backpropagates():
    model = NeuralEngineV0(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=4, router_depth=2,
        candidate_pool=8, active_circuits=2, internal_steps=3,
        typed_register_bridge=True,
    )
    inputs = torch.randint(1, 8, (4, 8))
    logits, stats = model(inputs, adaptive=False)

    assert logits.shape == (4, 64)
    assert stats["register_contexts"].shape == (4, 3, 32)
    logits.square().mean().backward()
    assert model.register_value_embedding.weight.grad is not None
    assert torch.isfinite(model.register_value_embedding.weight.grad).all()


def test_typed_register_bridge_straight_through_has_hard_forward_register():
    model = NeuralEngineV0(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=4, router_depth=2,
        candidate_pool=8, active_circuits=2, internal_steps=3,
        typed_register_bridge=True, register_bridge_mode="straight_through",
    )
    inputs = torch.randint(1, 8, (4, 8))
    _, stats = model(inputs, adaptive=False)
    probabilities = stats["register_probabilities"]
    assert probabilities.shape == (4, 3, 64)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(4, 3))
    assert bool(probabilities.eq(0).sum(dim=-1).eq(63).all())


def test_typed_register_bridge_can_preserve_multiple_intermediate_slots():
    model = NeuralEngineV0(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=4, router_depth=2,
        candidate_pool=8, active_circuits=2, internal_steps=3,
        typed_register_bridge=True, register_slot_count=2,
    )
    inputs = torch.randint(1, 8, (4, 8))
    _, stats = model(inputs, adaptive=False)

    assert stats["register_contexts"].shape == (4, 3, 32)
    assert stats["register_slot_contexts"].shape == (4, 3, 2, 32)
    assert model.parameter_report()["register_slot_count"] == 2
    loss = stats["register_contexts"].square().mean()
    loss.backward()
    assert model.register_value_embedding.weight.grad is not None
    assert torch.isfinite(model.register_value_embedding.weight.grad).all()


def test_typed_register_bridge_can_read_slots_with_structured_mixer():
    model = NeuralEngineV0(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=4, router_depth=2,
        candidate_pool=8, active_circuits=2, internal_steps=3,
        typed_register_bridge=True, register_slot_count=2,
        register_slot_read_mode="mix",
    )
    inputs = torch.randint(1, 8, (4, 8))
    _, stats = model(inputs, adaptive=False)

    assert stats["register_contexts"].shape == (4, 3, 32)
    assert model.parameter_report()["register_slot_read_mode"] == "mix"
    loss = stats["register_contexts"].square().mean()
    loss.backward()
    assert model.register_slot_mixer.weight.grad is not None
    assert torch.isfinite(model.register_slot_mixer.weight.grad).all()


def test_operation_transition_adapter_is_neutral_until_trained_and_has_gradients():
    model = NeuralEngineV0(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=4, router_depth=2,
        candidate_pool=8, active_circuits=2, internal_steps=3,
        operation_transition_rank=4,
    )
    batch = SyntheticTaskGenerator(seq_len=8, seed=72).batch(4)
    baseline = NeuralEngineV0(
        vocab_size=128, num_classes=64, seq_len=8, d_model=32, state_dim=32,
        num_circuits=64, circuit_rank=4, router_branch=4, router_depth=2,
        candidate_pool=8, active_circuits=2, internal_steps=3,
    )
    baseline.load_state_dict(model.state_dict(), strict=False)
    with torch.no_grad():
        transition_logits, _ = model(batch.inputs, adaptive=False)
        baseline_logits, _ = baseline(batch.inputs, adaptive=False)
    assert torch.allclose(transition_logits, baseline_logits)
    loss = torch.nn.functional.cross_entropy(model(batch.inputs)[0], batch.targets)
    loss.backward()
    assert model.operation_transition_up.grad is not None
    assert torch.isfinite(model.operation_transition_up.grad).all()


def test_serial_circuit_mode_has_same_shapes_and_gradients():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=2,
                           circuit_mode="serial")
    batch = SyntheticTaskGenerator(seed=9).batch(4)
    logits, _ = model(batch.inputs)
    torch.nn.functional.cross_entropy(logits, batch.targets).backward()
    assert logits.shape == (4, 64)
    assert model.circuits.down.grad is not None


def test_numeric_value_encoding_handles_unseen_value_tokens():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=2, slot_count=5,
                           numeric_value_encoding=True)
    low = torch.zeros(1, 32, dtype=torch.long)
    low[0, :3] = torch.tensor([1, 32 + 7, 32 + 11])
    high = low.clone()
    high[0, 1] = 32 + 39
    encoded = model.encode(high)
    assert encoded.shape == (1, 32)
    logits, _ = model(high)
    assert logits.shape == (1, 64)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0])).backward()
    assert model.value_encoder.weight.grad is not None


def test_adaptive_halting_skips_later_circuits():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=3,
                           adaptive_halting=True)
    model.halt_head.bias.data.fill_(10.0)
    batch = SyntheticTaskGenerator(seed=12).batch(4)
    with torch.no_grad():
        logits, stats = model(batch.inputs, adaptive=True)
        serving_logits, serving_stats = model(
            batch.inputs, adaptive=True, collect_stats=False)
    assert logits.shape == (4, 64)
    assert torch.allclose(logits, serving_logits)
    assert serving_stats == {}
    assert stats["executed_steps"].tolist() == [1, 1, 1, 1]
    assert bool(stats["selected_ids"][:, 1:].eq(-1).all())


def test_forced_route_replay_preserves_recorded_circuit_path():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=2)
    batch = SyntheticTaskGenerator(seed=13).batch(4)
    with torch.no_grad():
        _, original = model(batch.inputs, adaptive=False)
        replayed, replay_stats = model(
            batch.inputs, adaptive=False,
            forced_selected_ids=original["selected_ids"],
            forced_selected_weights=original["selected_weights"],
            forced_route_gains=original["route_gains"],
        )
    assert replayed.shape == (4, 64)
    assert torch.equal(replay_stats["selected_ids"], original["selected_ids"])


def test_forced_route_replay_allows_per_step_no_override_sentinel():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=2)
    batch = SyntheticTaskGenerator(seed=131).batch(4)
    with torch.no_grad():
        _, original = model(batch.inputs, adaptive=False)
        partial_ids = original["selected_ids"].clone()
        partial_ids[:, 0] = -1
        _, partial = model(batch.inputs, adaptive=False, forced_selected_ids=partial_ids)
    assert torch.equal(partial["selected_ids"][:, 1], original["selected_ids"][:, 1])
    assert bool(partial["selected_ids"][:, 0].ge(0).all())


def test_family_local_router_uses_semantic_task_family():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=16, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=3,
                           router_variant="family_local", family_count=2)
    batch = SyntheticTaskGenerator(seed=14).batch(4)
    with torch.no_grad():
        logits, stats = model(batch.inputs)
    assert logits.shape == (4, 64)
    assert stats["selected_ids"].shape == (4, 3, 2)
    assert isinstance(model.router, StableFamilyRouter)


def test_family_conditioned_router_keeps_global_bank():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=16, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=3,
                           router_variant="family_conditioned", family_count=2)
    batch = SyntheticTaskGenerator(seed=15).batch(4)
    with torch.no_grad():
        logits, stats = model(batch.inputs)
    assert logits.shape == (4, 64)
    assert stats["selected_ids"].shape == (4, 3, 2)
    assert model.router.num_circuits == 16


def test_semantic_family_mapping_splits_four_domains():
    model = NeuralEngineV0(num_circuits=32, state_dim=16, d_model=16,
                           circuit_rank=2, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=1,
                           family_count=4, router_variant="family_local")
    inputs = torch.zeros(4, 32, dtype=torch.long)
    inputs[:, 0] = torch.tensor([1, 4, 7, 10])
    assert torch.equal(model.semantic_family_ids(inputs), torch.tensor([0, 1, 2, 3]))
