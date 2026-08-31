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


def test_position_conditioning_preserves_operand_order():
    model = NeuralEngineV0(vocab_size=128, num_classes=64, seq_len=32, d_model=32, state_dim=32,
                           num_circuits=32, circuit_rank=4, router_branch=2, router_depth=2,
                           candidate_pool=4, active_circuits=2, internal_steps=2)
    first = torch.zeros(1, 32, dtype=torch.long)
    first[0, :3] = torch.tensor([1, 32 + 7, 32 + 11])
    second = first.clone()
    second[0, 1], second[0, 2] = second[0, 2], second[0, 1]
    assert not torch.allclose(model.encode(first), model.encode(second))
