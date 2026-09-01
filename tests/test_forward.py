import torch

from data.generator import SyntheticTaskGenerator
from neural_engine.model import NeuralEngineV0


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
    assert logits.shape == (4, 64)
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
