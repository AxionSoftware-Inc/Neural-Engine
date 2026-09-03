import torch

from data.dynamic_composition import DynamicCompositionGenerator
from neural_engine.dynamic_register import DynamicRegisterNeuralEngine
from neural_engine.modular_templates import TrainableModularTemplateRegister


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
