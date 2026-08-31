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
