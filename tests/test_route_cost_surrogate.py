import torch

from benchmark_route_cost_surrogate import _feature_tensor
from neural_engine.model import NeuralEngineV0


def test_output_logit_signature_uses_candidate_output_head():
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
    ).eval()
    _, stats = model(torch.randint(0, 32, (3, 4)), adaptive=False,
                    collect_stats=True)

    features = _feature_tensor(model, stats, step=0, signature_mode="output_logits")

    expected_dim = (model.state_dim * 3 + 1 + model.output[-1].out_features
                    + model.internal_steps)
    assert features.shape == (3, model.router.candidate_pool, expected_dim)
    assert torch.isfinite(features).all()
