import torch

from benchmark_target_retrieval_distill import _freeze_except_retriever
from neural_engine.model import NeuralEngineV0


def test_retrieval_distill_freezes_body_and_exposes_hierarchical_router():
    model = NeuralEngineV0(
        vocab_size=32,
        num_classes=5,
        seq_len=4,
        d_model=8,
        state_dim=8,
        num_circuits=16,
        circuit_rank=2,
        router_branch=2,
        router_depth=2,
        candidate_pool=4,
        active_circuits=2,
        internal_steps=2,
    )
    trainable = _freeze_except_retriever(model)

    assert trainable == [model.router.level_projections,
                         model.router.level_bias, model.router.keys]
    assert model.router.keys.requires_grad
    assert model.router.level_projections.requires_grad
    assert model.router.level_bias.requires_grad
    assert not model.circuits.down.requires_grad
    assert not model.state.update.weight_ih.requires_grad
