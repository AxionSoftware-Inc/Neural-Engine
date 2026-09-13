import torch
import pytest

from data.dynamic_composition import DynamicCompositionGenerator
from neural_engine.dynamic_register import DynamicRegisterNeuralEngine
from neural_engine.macro_growth import expand_macro_model
from neural_engine.modular_templates import TrainableModularTemplateRegister
from train_dynamic_composition import concatenate_batches, evaluate


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


def test_dynamic_generator_supports_non_modular_targets_with_offset():
    generator = DynamicCompositionGenerator(
        max_ops=4, train_max_ops=2, modulus=None, value_min=0, value_max=3,
        target_offset=64, split="all", seed=31,
    )
    batch = generator.balanced_batch(8)
    assert batch.targets.min().item() >= 0
    assert batch.targets.max().item() < 512
    assert batch.stage_targets.min().item() >= 0
    assert batch.stage_targets.max().item() < 512


def test_dynamic_generator_wide_non_modular_targets_fit_declared_head():
    generator = DynamicCompositionGenerator(
        max_ops=4, train_max_ops=2, modulus=None, value_min=0, value_max=7,
        target_offset=4096, split="heldout", seed=33,
    )
    batch = generator.balanced_batch(64)
    assert batch.targets.min().item() >= 0
    assert batch.targets.max().item() < 32768
    assert batch.stage_targets.min().item() >= 0
    assert batch.stage_targets.max().item() < 32768


def test_dynamic_generator_supports_weighted_value_range_mixture():
    generator = DynamicCompositionGenerator(
        max_ops=2,
        train_max_ops=2,
        modulus=None,
        value_min=0,
        value_max=95,
        value_ranges=((0, 7), (88, 95)),
        value_range_weights=(0.5, 0.5),
        seed=341,
    )
    batch = generator.batch(128)
    values = batch.inputs[:, 3:]
    values = values[values.ne(0)] - 32
    assert bool(((values <= 7) | (values >= 88)).all())
    assert bool((values <= 7).any())
    assert bool((values >= 88).any())


def test_dynamic_generator_can_share_one_value_range_per_program():
    generator = DynamicCompositionGenerator(
        max_ops=4,
        train_max_ops=4,
        modulus=None,
        value_min=0,
        value_max=95,
        value_ranges=((0, 7), (88, 95)),
        value_range_weights=(0.5, 0.5),
        value_range_mode="shared_per_program",
        seed=7,
    )
    batch = generator.task_balanced_batch(64)
    for row, depth in zip(batch.inputs, batch.depths):
        values = row[5 : 5 + int(depth) + 1] - 32
        assert bool((values.max() <= 7) or (values.min() >= 88))


def test_concatenate_batches_preserves_dynamic_composition_fields():
    generator = DynamicCompositionGenerator(
        max_ops=4, train_max_ops=4, modulus=None, value_min=0, value_max=3,
        seed=8,
    )
    first = generator.task_balanced_batch(2)
    second = generator.task_balanced_batch(3)
    combined = concatenate_batches(first, second)
    assert combined.inputs.shape[0] == 5
    assert combined.stage_targets.shape[0] == 5
    assert combined.stage_mask.shape[0] == 5


def test_dynamic_register_supports_non_modular_forward_without_modular_prior():
    model = DynamicRegisterNeuralEngine(
        max_ops=4, num_classes=512, modulus=None, seq_len=10,
        d_model=32, state_dim=32, num_circuits=64, circuit_rank=4,
        router_depth=2, candidate_pool=8, active_circuits=4, factor_count=8,
    )
    generator = DynamicCompositionGenerator(
        max_ops=4, train_max_ops=2, modulus=None, value_min=0, value_max=3,
        target_offset=64, seed=32,
    )
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 512)
    assert stats["step_logits"].shape == (4, 4, 512)
    assert model.parameter_report()["modulus"] is None


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


def test_factorized_serial_bmm_dispatch_matches_einsum():
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
        circuit_mode="serial",
    ).eval()
    generator = DynamicCompositionGenerator(max_ops=4, train_max_ops=4, seed=55)
    inputs = generator.task_balanced_batch(6).inputs
    with torch.no_grad():
        model.circuits.serial_dispatch = "einsum"
        reference, _ = model(inputs)
        model.circuits.serial_dispatch = "bmm"
        optimized, _ = model(inputs)
    assert torch.allclose(reference, optimized, atol=1e-5, rtol=1e-5)


def test_typed_digit_authoritative_output_is_opt_in_and_reported():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        num_classes=16 ** 2,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        output_mode="factorized_digits",
        output_digit_base=16,
        output_digit_count=2,
        typed_digit_state=True,
        typed_digit_dim=16,
        typed_digit_base=16,
        typed_digit_count=2,
        typed_digit_carry_chain=True,
        typed_digit_output_authoritative=True,
    )
    generator = DynamicCompositionGenerator(
        max_ops=2, train_max_ops=2, modulus=None, value_min=0, value_max=7, seed=41,
    )
    _, stats = model(generator.batch(3).inputs)
    assert model.parameter_report()["typed_digit_output_authoritative"] is True
    assert len(stats["digit_logits"]) == 2


def test_typed_digit_write_residual_is_opt_in_and_multiply_scoped():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        num_classes=16 ** 2,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        output_mode="factorized_digits",
        output_digit_base=16,
        output_digit_count=2,
        typed_digit_state=True,
        typed_digit_dim=16,
        typed_digit_base=16,
        typed_digit_count=2,
        typed_digit_carry_chain=True,
        typed_digit_write_scale=0.1,
        typed_digit_write_multiply_only=True,
    )
    assert model.parameter_report()["typed_digit_write_scale"] == 0.1
    assert model.parameter_report()["typed_digit_write_multiply_only"] is True


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


def test_dynamic_register_fixed_fourier_value_encoder_has_no_trainable_projection():
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
        value_encoder_mode="fixed_fourier",
    )
    assert not list(model.value_encoder.parameters())
    assert model.parameter_report()["value_encoder_mode"] == "fixed_fourier"


def test_dynamic_register_nonmod_value_encoder_can_represent_values_above_63():
    model = DynamicRegisterNeuralEngine(
        max_ops=1,
        seq_len=4,
        d_model=32,
        state_dim=32,
        num_circuits=16,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=4,
        modulus=None,
        value_encoder_modulus=128,
    )
    inputs = torch.tensor([
        [1, 2, 96, 96],   # raw operand 64
        [1, 2, 127, 127], # raw operand 95
    ])
    encoded = model.encode_program(inputs)
    assert model.parameter_report()["value_encoder_modulus"] == 128
    assert not torch.allclose(encoded[0, 2], encoded[1, 2])


def test_dynamic_register_hybrid_fourier_value_encoder_keeps_learned_projection():
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
        value_encoder_mode="hybrid_fourier",
    )
    assert list(model.value_encoder.parameters())
    assert model.parameter_report()["value_encoder_mode"] == "hybrid_fourier"


def test_dynamic_register_shared_factor_mix_has_constant_mix_storage():
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
        factor_mix_mode="shared",
    )
    assert tuple(model.circuits.factor_mix.shape) == (2,)


def test_dynamic_register_geometric_digit_output_is_ordered_and_compact():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        num_classes=512 ** 3 * 2,
        output_mode="factorized_digits",
        output_digit_base=512,
        output_digit_count=4,
        output_factor_rank=8,
        output_digit_geometry=True,
        output_digit_temperature=2.0,
    )
    states = torch.zeros(3, 32)
    digits = model.output[1].digit_logits(states)
    assert len(digits) == 4
    assert tuple(digits[0].shape) == (3, 2)
    assert tuple(digits[1].shape) == (3, 512)
    assert torch.isfinite(digits[-1]).all()
    assert model.parameter_report()["output_digit_geometry"] is True


def test_dynamic_register_operation_step_routing_is_value_independent():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        route_context_mode="operation_step",
        route_exploration_prob=0.0,
    )
    batch = torch.tensor([
        [1, 2, 3, 32, 33, 0, 0, 0],
        [1, 2, 3, 62, 63, 0, 0, 0],
    ])
    model.eval()
    _, stats = model(batch)
    assert torch.equal(stats["selected_ids"][0], stats["selected_ids"][1])


def test_dynamic_register_hybrid_routing_keeps_value_dependence():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        route_context_mode="hybrid",
        route_exploration_prob=0.0,
    )
    batch = torch.tensor([
        [1, 2, 3, 32, 33, 0, 0, 0],
        [1, 2, 3, 62, 63, 0, 0, 0],
    ])
    model.eval()
    _, stats = model(batch)
    assert not torch.equal(stats["selected_ids"][0], stats["selected_ids"][1])
    assert model.parameter_report()["route_context_mode"] == "hybrid"


def test_dynamic_register_modular_prior_tracks_fixed_transition():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        modular_prior=True,
    )
    batch = torch.tensor([[1, 2, 3, 32, 33, 0, 0, 0]])
    _, stats = model(batch)
    assert tuple(model.modular_transition.shape) == (3, 64, 64)
    assert stats["step_logits"].shape == (1, 2, 64)
    assert model.parameter_report()["modular_prior"] is True


def test_dynamic_register_template_prior_has_no_transition_table():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        modular_prior=True,
        modular_prior_mode="templates",
    )
    assert not hasattr(model, "modular_transition")
    assert tuple(model.modular_template_logits.shape) == (3, 3)
    batch = torch.tensor([[1, 2, 3, 32, 33, 0, 0, 0]])
    logits, _ = model(batch)
    assert logits.shape == (1, 64)
    assert model.parameter_report()["modular_prior_mode"] == "templates"


def test_trainable_modular_templates_keep_no_dense_transition_table():
    model = TrainableModularTemplateRegister(max_ops=2)
    batch = torch.tensor([[1, 2, 3, 32, 33, 0, 0, 0]])
    logits, stats = model(batch)
    assert logits.shape == (1, 64)
    assert stats["step_logits"].shape == (1, 2, 64)
    assert model.parameter_report()["dense_transition_table"] is False
    assert model.parameter_report()["total_params"] < 10_000


def test_trainable_modular_templates_support_another_modulus_and_random_init():
    model = TrainableModularTemplateRegister(
        max_ops=2, num_classes=32, modulus=32, template_init="random"
    )
    batch = torch.tensor([[1, 2, 3, 32, 33, 0, 0, 0]])
    logits, _ = model(batch)
    assert logits.shape == (1, 32)
    assert model.parameter_report()["modulus"] == 32
    assert model.parameter_report()["template_init"] == "random"


def test_dynamic_register_template_prior_supports_another_modulus():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        num_classes=32,
        modulus=32,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        modular_prior=True,
        modular_prior_mode="templates",
        modular_template_init="random",
    )
    generator = DynamicCompositionGenerator(
        max_ops=2, train_max_ops=2, modulus=32, value_max=31, seed=13
    )
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 32)
    assert tuple(model.modular_template_logits.shape) == (3, 3)
    assert model.parameter_report()["modulus"] == 32


def test_dynamic_register_can_disable_circuit_residual_path():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        modular_prior=True,
        modular_prior_mode="templates",
        circuit_residual_scale=0.0,
    )
    batch = torch.tensor([[1, 2, 3, 32, 33, 0, 0, 0]])
    logits, _ = model(batch)
    assert logits.shape == (1, 64)
    assert model.parameter_report()["circuit_residual_scale"] == 0.0


def test_dynamic_register_operation_adapter_is_shared_by_step():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        factor_mix_mode="shared",
        operation_adapter_rank=4,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=14)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.operation_adapter_down.shape == (3, 32, 4)
    assert model.parameter_report()["operation_adapter_rank"] == 4


def test_dynamic_register_operation_adapter_gate_starts_closed():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        factor_mix_mode="shared",
        operation_adapter_rank=4,
        operation_adapter_gate=True,
    )
    assert model.operation_adapter_gate.item() == 0.0
    assert model.parameter_report()["operation_adapter_gate"] is True


def test_dynamic_register_operation_write_adapter_is_operation_conditioned():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        operation_write_adapter_rank=4,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=15)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.operation_write_adapter_down.shape == (3, 32, 4)
    assert model.parameter_report()["operation_write_adapter_rank"] == 4
    assert model.parameter_report()["operation_write_adapter_mode"] == "post_state"


def test_dynamic_register_pre_writer_adapter_changes_write_locus():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        operation_write_adapter_rank=4,
        operation_write_adapter_mode="pre_writer",
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=151)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.parameter_report()["operation_write_adapter_mode"] == "pre_writer"


def test_dynamic_register_terminal_adapter_preserves_intermediate_state_path():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        operation_write_adapter_rank=4,
        operation_write_adapter_mode="terminal_only",
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=152)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.parameter_report()["operation_write_adapter_mode"] == "terminal_only"


def test_dynamic_register_structured_numeric_state_is_optional_and_recurrent():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        numeric_state_dim=8,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=16)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.parameter_report()["numeric_state_dim"] == 8


def test_dynamic_register_typed_teacher_forcing_is_opt_in_and_shape_safe():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_classes=4096,
        modulus=None,
        circuit_rank=4,
        num_circuits=64,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        typed_digit_state=True,
        typed_digit_dim=4,
        typed_digit_base=16,
        typed_digit_count=3,
        typed_digit_value_offset=64,
        typed_digit_carry_chain=True,
        output_mode="factorized_digits",
        output_digit_base=16,
        output_digit_count=3,
    )
    generator = DynamicCompositionGenerator(
        max_ops=2,
        train_max_ops=2,
        modulus=None,
        value_min=0,
        value_max=3,
        target_offset=64,
        seed=162,
    )
    batch = generator.batch(4)
    logits, stats = model(
        batch.inputs,
        teacher_stage_targets=batch.stage_targets - 64,
        teacher_forcing_probability=1.0,
    )
    assert logits.shape == (4, 4096)
    assert len(stats["typed_digit_logits"]) == 3


def test_dynamic_register_typed_multiply_convolution_is_opt_in():
    model = DynamicRegisterNeuralEngine(
        vocab_size=128,
        num_classes=4**4,
        max_ops=2,
        seq_len=1 + 2 + 3,
        d_model=32,
        state_dim=32,
        num_circuits=16,
        circuit_rank=4,
        router_branch=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=2,
        typed_digit_state=True,
        typed_digit_dim=4,
        typed_digit_base=4,
        typed_digit_count=4,
        typed_digit_carry_chain=True,
        typed_digit_multiply_convolution=True,
        output_mode="factorized_digits",
        output_digit_base=4,
        output_digit_count=4,
        output_factor_rank=8,
    )
    assert model.typed_digit_multiply_transition is not None
    assert model.parameter_report()["typed_digit_multiply_convolution"] is True
    inputs = torch.tensor([
        [1, 4, 2, 32, 33, 34],
        [1, 2, 4, 35, 36, 0],
    ])
    logits, stats = model(inputs, return_full_logits=True)
    assert logits.shape == (2, 4**4)
    assert torch.isfinite(logits).all()
    assert stats["typed_digit_logits"][0].shape[:2] == (2, 2)


def test_dynamic_register_rejects_multiply_convolution_without_typed_carry():
    with pytest.raises(ValueError, match="requires typed carry state"):
        DynamicRegisterNeuralEngine(
            vocab_size=128,
            num_classes=2**8,
            max_ops=1,
            seq_len=1 + 1 + 2,
            d_model=16,
            state_dim=16,
            num_circuits=8,
            circuit_rank=2,
            router_branch=2,
            router_depth=1,
            candidate_pool=4,
            active_circuits=1,
            typed_digit_multiply_convolution=True,
        )


def test_dynamic_register_typed_numeric_multiply_convolution_is_opt_in():
    model = DynamicRegisterNeuralEngine(
        vocab_size=128,
        num_classes=4**4,
        max_ops=2,
        seq_len=1 + 2 + 3,
        d_model=32,
        state_dim=32,
        num_circuits=16,
        circuit_rank=4,
        router_branch=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=2,
        typed_digit_state=True,
        typed_digit_dim=4,
        typed_digit_base=4,
        typed_digit_count=4,
        typed_digit_carry_chain=True,
        typed_digit_multiply_numeric_convolution=True,
        output_mode="factorized_digits",
        output_digit_base=4,
        output_digit_count=4,
        output_factor_rank=8,
    )
    assert model.typed_digit_multiply_transition is not None
    assert model.typed_digit_multiply_numeric_projection is not None
    assert model.parameter_report()["typed_digit_multiply_numeric_convolution"] is True
    inputs = torch.tensor([
        [1, 4, 2, 32, 33, 34],
        [1, 2, 4, 35, 36, 0],
    ])
    logits, stats = model(inputs)
    assert logits.shape == (2, 4**4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(stats["typed_digit_logits"][0]).all()


def test_dynamic_register_typed_multiply_pair_table_is_opt_in():
    model = DynamicRegisterNeuralEngine(
        vocab_size=128,
        num_classes=4**4,
        max_ops=2,
        seq_len=1 + 2 + 3,
        d_model=32,
        state_dim=32,
        num_circuits=16,
        circuit_rank=4,
        router_branch=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=2,
        typed_digit_state=True,
        typed_digit_dim=4,
        typed_digit_base=4,
        typed_digit_count=4,
        typed_digit_carry_chain=True,
        typed_digit_multiply_pair_table=True,
        output_mode="factorized_digits",
        output_digit_base=4,
        output_digit_count=4,
        output_factor_rank=8,
    )
    assert model.typed_digit_multiply_transition is not None
    assert model.typed_digit_multiply_pair_tables is not None
    assert model.parameter_report()["typed_digit_multiply_pair_table"] is True
    inputs = torch.tensor([
        [1, 4, 2, 32, 33, 34],
        [1, 2, 4, 35, 36, 0],
    ])
    logits, stats = model(inputs)
    assert logits.shape == (2, 4**4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(stats["typed_digit_logits"][0]).all()


def test_dynamic_register_typed_output_multiply_only_requires_authoritative():
    with pytest.raises(ValueError, match="requires authoritative typed output"):
        DynamicRegisterNeuralEngine(
            max_ops=2,
            seq_len=8,
            d_model=16,
            state_dim=16,
            num_circuits=32,
            circuit_rank=2,
            router_depth=2,
            candidate_pool=4,
            active_circuits=2,
            factor_count=6,
            modulus=None,
            typed_digit_output_multiply_only=True,
        )


def test_dynamic_register_typed_output_multiply_only_is_opt_in():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        num_classes=16**2,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        modulus=None,
        typed_digit_state=True,
        typed_digit_base=16,
        typed_digit_count=2,
        typed_digit_carry_chain=True,
        typed_digit_output_authoritative=True,
        typed_digit_output_multiply_only=True,
        output_mode="factorized_digits",
        output_digit_base=16,
        output_digit_count=2,
    )
    assert model.parameter_report()["typed_digit_output_multiply_only"] is True
    generator = DynamicCompositionGenerator(
        max_ops=2,
        train_max_ops=2,
        value_min=0,
        value_max=3,
        seed=3453,
    )
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 16**2)


def test_dynamic_register_algebraic_state_double_preserves_precision_path():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        modulus=None,
        algebraic_state_mode="polynomial2_fourier",
        algebraic_state_double=True,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=3454)
    logits, stats = model(generator.batch(4).inputs, collect_state_stats=True)
    assert logits.shape == (4, 64)
    assert stats["algebraic_state_features"].dtype == torch.float64
    assert torch.isfinite(logits).all()
    assert model.parameter_report()["algebraic_state_double"] is True


def test_dynamic_register_algebraic_fourier_ladder_expands_digit_periods():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        modulus=None,
        output_mode="factorized_digits",
        output_digit_base=4,
        output_digit_count=3,
        algebraic_state_mode="polynomial2_fourier",
        algebraic_state_fourier_base=4,
        algebraic_state_fourier_ladder=True,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=3455)
    logits, stats = model(generator.batch(4).inputs, collect_state_stats=True)
    assert logits.shape == (4, 4**3)
    features = model._algebraic_state_features(stats["algebraic_state_features"][:, 0])
    assert features.shape[-1] == 2 + 3 * 2 * 7
    assert model.parameter_report()["algebraic_state_fourier_ladder"] is True


def test_dynamic_register_structured_scalar_state_has_shared_value_format():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        structured_scalar_state=True,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=161)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.structured_scalar_transition.shape == (3, 4)
    assert model.parameter_report()["structured_scalar_state"] is True
    assert model(generator.batch(4).inputs)[1]["structured_scalar_states"].shape == (4, 2)


def test_dynamic_register_operation_transition_conditions_write_input():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        operation_transition_rank=4,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.operation_transition_down.shape == (3, 32, 4)
    assert model.parameter_report()["operation_transition_rank"] == 4


def test_dynamic_register_bilinear_transition_conditions_write_input():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        operation_bilinear_transition_rank=4,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.operation_bilinear_acc_down.shape == (3, 32, 4)
    assert model.operation_bilinear_operand_down.shape == (3, 32, 4)
    assert model.parameter_report()["operation_bilinear_transition_rank"] == 4


def test_dynamic_register_scalar_gaussian_output_preserves_class_shape():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_classes=128,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        output_mode="scalar_gaussian",
        output_temperature=8.0,
        output_scalar_bias=16.0,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 128)
    assert model.output[-1].out_features == 128
    assert model.parameter_report()["output_mode"] == "scalar_gaussian"


def test_dynamic_register_operation_circuit_banks_select_by_operation():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        operation_circuit_bank=True,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert len(model.circuits) == 3
    assert stats["selected_ids"].shape == (4, 2, 4)
    assert model.parameter_report()["operation_circuit_bank"] is True


def test_factorized_router_can_warm_up_with_a_prefix_of_factor_capacity():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        factor_candidate_pool=4,
        factor_capacity=4,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    _, stats = model(generator.batch(4).inputs)
    selected_ids = stats["selected_ids"].reshape(-1)
    selected_ids = selected_ids[selected_ids.ge(0)]
    first, second = model.router._factor_ids(selected_ids)
    assert int(first.max()) < 4
    assert int(second.max()) < 4
    assert model.parameter_report()["factor_capacity"] == 4

    model.router.set_routing_state(factor_capacity=8)
    assert model.router.factor_capacity == 8
    assert model.parameter_report()["factor_capacity"] == 8


def test_dynamic_register_ordered_factor_slots_keep_pair_order_expressive():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        ordered_factor_slots=True,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert stats["selected_ids"].shape == (4, 2, 4)
    assert model.circuits.ordered_factor_slots is True
    assert model.parameter_report()["ordered_factor_slots"] is True


def test_dynamic_register_query_conditioned_factor_mix_is_sparse_and_recurrent():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        query_factor_mix_scale=0.5,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert stats["selected_ids"].shape == (4, 2, 4)
    assert model.circuits.factor_gate_keys.shape == (8, 32)
    assert model.parameter_report()["query_factor_mix_scale"] == 0.5


def test_dynamic_register_factor_pair_basis_adds_nonlinear_pair_capacity():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        factor_pair_rank=4,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert stats["selected_ids"].shape == (4, 2, 4)
    assert model.circuits.pair_codes.shape == (8, 4)
    assert model.parameter_report()["factor_pair_rank"] == 4


def test_dynamic_register_operation_router_keys_start_from_shared_geometry():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        operation_router_keys=True,
    )
    assert tuple(model.router.operation_key_deltas.shape) == (3, 8, 32)
    assert torch.count_nonzero(model.router.operation_key_deltas) == 0
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert stats["selected_ids"].shape == (4, 2, 4)
    assert model.parameter_report()["operation_router_keys"] is True


def test_dynamic_register_operation_read_adapter_is_optional_and_recurrent():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        operation_read_adapter_rank=4,
    )
    assert tuple(model.operation_read_adapter_down.shape) == (3, 32, 4)
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert stats["executed_mask"].any()
    assert model.parameter_report()["operation_read_adapter_rank"] == 4


def test_dynamic_register_predecessor_operation_context_has_start_token():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        predecessor_operation_context=True,
    )
    assert tuple(model.predecessor_operation_embedding.weight.shape) == (4, 32)
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert stats["executed_mask"].any()
    assert model.parameter_report()["predecessor_operation_context"] is True


def test_dynamic_register_dual_slot_state_uses_separate_writers():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        state_layout="dual_slot",
    )
    assert len(model.slot_writers) == 2
    assert not hasattr(model, "register_writer")
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert stats["executed_mask"].any()
    assert model.parameter_report()["state_layout"] == "dual_slot"


def test_dynamic_register_circuit_input_norm_is_optional():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        operation_circuit_bank=True,
        circuit_input_norm=True,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=17)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.circuit_input_norm is not None
    assert model.parameter_report()["circuit_input_norm"] is True


def test_dynamic_register_factorized_digit_output_reconstructs_classes():
    model = DynamicRegisterNeuralEngine(
        max_ops=4,
        num_classes=32768,
        modulus=None,
        seq_len=10,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        output_mode="factorized_digits",
        output_digit_base=128,
    )
    generator = DynamicCompositionGenerator(
        max_ops=4,
        train_max_ops=2,
        modulus=None,
        value_min=0,
        value_max=7,
        target_offset=4096,
        seed=34,
    )
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 32768)
    assert stats["digit_high_logits"].shape == (4, 4, 256)
    assert stats["digit_low_logits"].shape == (4, 4, 128)
    assert model.parameter_report()["output_mode"] == "factorized_digits"
    assert model.parameter_report()["output_digit_base"] == 128


def test_dynamic_register_factorized_digit_output_supports_shared_low_rank_codec():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        num_classes=32768,
        modulus=None,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        output_mode="factorized_digits",
        output_digit_base=128,
        output_factor_rank=8,
    )
    generator = DynamicCompositionGenerator(
        max_ops=2,
        train_max_ops=1,
        modulus=None,
        value_min=0,
        value_max=3,
        target_offset=64,
        seed=342,
    )
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 32768)
    assert stats["digit_high_logits"].shape == (4, 2, 256)
    assert stats["digit_low_logits"].shape == (4, 2, 128)
    assert model.parameter_report()["output_factor_rank"] == 8


def test_dynamic_register_factorized_digit_output_supports_three_digit_codec():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        num_classes=4096,
        modulus=None,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        output_mode="factorized_digits",
        output_digit_base=16,
        output_digit_count=3,
        output_factor_rank=8,
    )
    generator = DynamicCompositionGenerator(
        max_ops=2,
        train_max_ops=1,
        modulus=None,
        value_min=0,
        value_max=3,
        target_offset=64,
        seed=343,
    )
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 4096)
    assert len(stats["digit_logits"]) == 3
    assert all(digit.shape == (4, 2, 16) for digit in stats["digit_logits"])
    assert model.parameter_report()["output_digit_count"] == 3


def test_dynamic_register_factorized_digit_output_supports_cross_digit_context():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        num_classes=4096,
        modulus=None,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        output_mode="factorized_digits",
        output_digit_base=16,
        output_digit_count=3,
        output_factor_rank=8,
        output_digit_interaction_rank=4,
    )
    generator = DynamicCompositionGenerator(
        max_ops=2,
        train_max_ops=1,
        modulus=None,
        value_min=0,
        value_max=3,
        target_offset=64,
        seed=344,
    )
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 4096)
    assert len(stats["digit_logits"]) == 3
    assert model.parameter_report()["output_digit_interaction_rank"] == 4


def test_dynamic_register_factorized_digit_output_supports_straight_through_hard_context():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        num_classes=4096,
        modulus=None,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        output_mode="factorized_digits",
        output_digit_base=16,
        output_digit_count=3,
        output_factor_rank=8,
        output_digit_interaction_rank=4,
        output_digit_context_mode="straight_through_hard",
    )
    generator = DynamicCompositionGenerator(
        max_ops=2,
        train_max_ops=1,
        modulus=None,
        value_min=0,
        value_max=3,
        target_offset=64,
        seed=345,
    )
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 4096)
    assert model.parameter_report()["output_digit_context_mode"] == "straight_through_hard"


def test_dynamic_register_typed_digit_carry_chain_updates_all_slots():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        num_classes=4096,
        modulus=None,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        output_mode="factorized_digits",
        output_digit_base=16,
        output_digit_count=3,
        output_factor_rank=8,
        typed_digit_state=True,
        typed_digit_dim=4,
        typed_digit_base=16,
        typed_digit_count=3,
        typed_digit_value_offset=64,
        typed_digit_carry_chain=True,
    )
    generator = DynamicCompositionGenerator(
        max_ops=2,
        train_max_ops=1,
        modulus=None,
        value_min=0,
        value_max=3,
        target_offset=64,
        seed=346,
    )
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 4096)
    assert len(stats["typed_digit_logits"]) == 3
    assert all(digit.shape == (4, 2, 16) for digit in stats["typed_digit_logits"])
    report = model.parameter_report()
    assert report["typed_digit_state"] is True
    assert report["typed_digit_carry_chain"] is True


def test_dynamic_register_can_bridge_existing_algebraic_packet_to_output():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        num_classes=4096,
        modulus=None,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        output_mode="factorized_digits",
        output_digit_base=16,
        output_digit_count=3,
        output_factor_rank=8,
        algebraic_state_mode="polynomial2_fourier",
        algebraic_output_bridge_scale=1.0,
    )
    generator = DynamicCompositionGenerator(
        max_ops=2,
        train_max_ops=1,
        modulus=None,
        value_min=0,
        value_max=3,
        target_offset=64,
        seed=347,
    )
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 4096)
    assert model.parameter_report()["algebraic_output_bridge_scale"] == 1.0


def test_dynamic_register_operation_output_adapter_conditions_readout_on_last_op():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        num_classes=4096,
        modulus=None,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        output_mode="factorized_digits",
        output_digit_base=16,
        output_digit_count=3,
        output_factor_rank=8,
        operation_output_adapter_rank=4,
    )
    generator = DynamicCompositionGenerator(
        max_ops=2,
        train_max_ops=1,
        modulus=None,
        value_min=0,
        value_max=3,
        target_offset=64,
        seed=346,
    )
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 4096)
    assert stats["digit_logits"][0].shape == (4, 2, 16)
    assert model.parameter_report()["operation_output_adapter_rank"] == 4


def test_dynamic_register_can_skip_full_factorized_logits_during_training():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        num_classes=32768,
        modulus=None,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        output_mode="factorized_digits",
        output_digit_base=128,
    )
    generator = DynamicCompositionGenerator(
        max_ops=2,
        train_max_ops=2,
        modulus=None,
        value_min=0,
        value_max=3,
        target_offset=64,
        seed=341,
    )
    compact_logits, stats = model(
        generator.batch(4).inputs, return_full_logits=False
    )
    assert compact_logits.shape == (4, 256)
    assert stats["step_logits"] is None
    assert stats["digit_high_logits"].shape == (4, 2, 256)


def test_dynamic_composition_compact_factorized_evaluation_keeps_exact_argmax():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        num_classes=32768,
        modulus=None,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        output_mode="factorized_digits",
        output_digit_base=128,
    )
    generator = DynamicCompositionGenerator(
        max_ops=2,
        train_max_ops=1,
        modulus=None,
        value_min=0,
        value_max=3,
        target_offset=64,
        split="heldout",
        seed=342,
    )
    report = evaluate(model, generator, torch.device("cpu"), 2, compact_factorized=True)
    assert report["loss_mode"] == "factorized_digit_sum_compact"
    assert 0.0 <= report["accuracy"] <= 1.0


def test_dynamic_register_structured_scalar_read_path_is_optional():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        structured_scalar_state=True,
        structured_scalar_read_scale=1.0,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=36)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.parameter_report()["structured_scalar_read_scale"] == 1.0


def test_dynamic_register_authoritative_scalar_requires_and_uses_value_lane():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        structured_scalar_state=True,
        structured_scalar_authoritative=True,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=37)
    logits, stats = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert stats["step_logits"].shape == (4, 2, 64)
    assert model.parameter_report()["structured_scalar_authoritative"] is True


def test_dynamic_register_polynomial_algebraic_state_tracks_exact_composition():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        modulus=None,
        algebraic_state_mode="polynomial2",
        algebraic_state_value_scale=16.0,
    )
    # x=1; add 2 -> 3; multiply by 3 -> 9.
    inputs = torch.tensor([[1, 2, 4, 33, 34, 35, 0, 0]])
    _, stats = model(inputs, collect_state_stats=True)
    features = stats["algebraic_state_features"][0]
    expected = torch.tensor([[3.0 / 16.0, 9.0 / 256.0], [9.0 / 16.0, 81.0 / 256.0]])
    assert torch.allclose(features, expected, atol=1e-6)
    assert model.parameter_report()["algebraic_state_mode"] == "polynomial2"


def test_dynamic_register_fourier_algebraic_bridge_adds_range_features():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        modulus=None,
        algebraic_state_mode="polynomial2_fourier",
        algebraic_state_value_scale=16.0,
    )
    inputs = torch.tensor([[1, 2, 4, 33, 34, 35, 0, 0]])
    _, stats = model(inputs, collect_state_stats=True)
    assert stats["algebraic_state_features"].shape == (1, 2, 2)
    assert model.algebraic_state_projection[0].in_features == 44
    assert model.parameter_report()["algebraic_state_mode"] == "polynomial2_fourier"


def test_dynamic_register_algebraic_write_bridge_is_opt_in():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        modulus=None,
        algebraic_state_mode="polynomial2",
        algebraic_state_write_scale=1.0,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=343)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.parameter_report()["algebraic_state_write_scale"] == 1.0


def test_dynamic_register_algebraic_authoritative_read_is_opt_in():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        modulus=None,
        algebraic_state_mode="polynomial2",
        algebraic_state_authoritative_read=True,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=344)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.parameter_report()["algebraic_state_authoritative_read"] is True


def test_dynamic_register_algebraic_output_decoder_is_opt_in():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        modulus=None,
        algebraic_state_mode="polynomial2_fourier",
        algebraic_output_decoder=True,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=345)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.parameter_report()["algebraic_output_decoder"] is True


def test_dynamic_register_operation_conditioned_algebraic_projection_is_opt_in():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        modulus=None,
        algebraic_state_mode="polynomial2_fourier",
        algebraic_state_operation_conditioned=True,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=3451)
    logits, _ = model(generator.batch(6).inputs)
    assert logits.shape == (6, 64)
    assert model.parameter_report()["algebraic_state_operation_conditioned"] is True


def test_dynamic_register_multiply_algebraic_residual_is_opt_in():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        modulus=None,
        algebraic_state_mode="polynomial2_fourier",
        algebraic_state_multiply_residual=True,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=3452)
    logits, _ = model(generator.batch(6).inputs)
    assert logits.shape == (6, 64)
    assert model.parameter_report()["algebraic_state_multiply_residual"] is True


def test_dynamic_register_exact_integer_output_decoder_is_opt_in():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        modulus=None,
        algebraic_state_mode="polynomial2",
        output_mode="factorized_digits",
        output_digit_base=8,
        algebraic_integer_output_decoder=True,
        algebraic_integer_digit_dim=8,
        algebraic_integer_output_decoder_mode="multiply_only",
        algebraic_integer_output_head=True,
        algebraic_integer_output_factor_rank=4,
        algebraic_integer_output_digit_interaction_rank=2,
        algebraic_integer_state_read_scale=0.125,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=346)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    assert model.parameter_report()["algebraic_integer_output_decoder"] is True
    assert model.parameter_report()["algebraic_integer_output_decoder_mode"] == "multiply_only"
    assert model.parameter_report()["algebraic_integer_output_head"] is True
    assert model.parameter_report()["algebraic_integer_output_factor_rank"] == 4
    assert model.parameter_report()["algebraic_integer_state_read_scale"] == 0.125


def test_dynamic_register_exact_integer_output_head_can_cover_all_operations():
    model = DynamicRegisterNeuralEngine(
        vocab_size=128,
        num_classes=512 ** 4,
        modulus=None,
        max_ops=2,
        d_model=32,
        state_dim=32,
        num_circuits=8,
        circuit_rank=4,
        router_branch=2,
        router_depth=1,
        candidate_pool=4,
        active_circuits=2,
        algebraic_state_mode="polynomial2",
        algebraic_integer_output_decoder=True,
        algebraic_integer_output_decoder_mode="all",
        algebraic_integer_output_head=True,
        algebraic_integer_output_factor_rank=4,
        algebraic_integer_output_digit_interaction_rank=2,
        output_mode="factorized_digits",
        output_digit_base=512,
        output_digit_count=4,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=347)
    logits, stats = model(generator.batch(4).inputs, return_full_logits=False)
    assert logits.shape == (4, 512)
    assert len(stats["digit_logits"]) == 4


def test_dynamic_register_can_collect_recurrent_state_trace():
    model = DynamicRegisterNeuralEngine(
        max_ops=3,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
    )
    generator = DynamicCompositionGenerator(max_ops=3, train_max_ops=2, seed=35)
    logits, stats = model(generator.batch(5).inputs, collect_state_stats=True)
    assert logits.shape == (5, 64)
    for name in (
        "pre_accumulator_states",
        "query_states",
        "post_accumulator_states",
        "step_states",
    ):
        assert stats[name].shape == (5, 3, 16)


def test_dynamic_register_residual_state_update_preserves_explicit_mode():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=32,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=6,
        state_update_mode="residual",
        state_residual_scale=0.25,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=38)
    logits, _ = model(generator.batch(4).inputs)
    assert logits.shape == (4, 64)
    report = model.parameter_report()
    assert report["state_update_mode"] == "residual"
    assert report["state_residual_scale"] == 0.25


def test_dynamic_register_macro_cells_add_sparse_multi_step_path():
    model = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=32,
        state_dim=32,
        num_circuits=64,
        circuit_rank=4,
        router_depth=2,
        candidate_pool=8,
        active_circuits=4,
        factor_count=8,
        macro_cell_count=16,
        macro_cell_rank=4,
        macro_cell_depth=2,
        macro_candidate_pool=4,
        active_macro_cells=1,
    )
    generator = DynamicCompositionGenerator(max_ops=2, train_max_ops=2, seed=11)
    batch = generator.task_balanced_batch(6)
    logits, stats = model(batch.inputs)
    assert logits.shape == (6, 64)
    assert tuple(stats["macro_selected_ids"].shape) == (6, 2, 1)
    assert tuple(stats["macro_selected_weights"].shape) == (6, 2, 1)
    report = model.parameter_report()
    assert report["macro_cell_count"] == 16
    assert report["active_macro_cells"] == 1
    assert report["macro_total_params"] > report["macro_active_params_estimate"]


def test_macro_growth_preserves_parent_rows_and_opens_new_router_levels():
    parent = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=16,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=4,
        macro_cell_count=16,
        macro_cell_rank=2,
        macro_cell_depth=2,
        macro_router_branch=4,
        macro_candidate_pool=4,
        active_macro_cells=1,
    )
    grown = DynamicRegisterNeuralEngine(
        max_ops=2,
        seq_len=8,
        d_model=16,
        state_dim=16,
        num_circuits=16,
        circuit_rank=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        factor_count=4,
        macro_cell_count=256,
        macro_cell_rank=2,
        macro_cell_depth=2,
        macro_router_branch=4,
        macro_candidate_pool=4,
        active_macro_cells=1,
    )
    with torch.no_grad():
        parent.macro_cell_bank.down.fill_(3.0)
        parent.macro_router.level_projections.fill_(5.0)
        parent.macro_router.level_bias.fill_(6.0)
    expand_macro_model(parent, grown)
    assert torch.equal(grown.macro_cell_bank.down[:16], parent.macro_cell_bank.down)
    assert torch.equal(
        grown.macro_router.level_projections[:, :2],
        parent.macro_router.level_projections,
    )
    assert torch.equal(
        grown.macro_router.level_projections[:, 2:],
        torch.zeros_like(grown.macro_router.level_projections[:, 2:]),
    )
    assert torch.equal(
        grown.macro_router.level_bias[:, 2:],
        torch.zeros_like(grown.macro_router.level_bias[:, 2:]),
    )
