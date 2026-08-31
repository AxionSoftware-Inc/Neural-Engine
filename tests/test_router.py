import torch

from neural_engine.router import HierarchicalRouter


def test_router_returns_local_structured_selection():
    router = HierarchicalRouter(32, num_circuits=128, branch=4, depth=3, candidate_pool=16, active_circuits=4)
    selected, weights, stats = router(torch.randn(7, 32))
    assert selected.shape == (7, 4)
    assert weights.shape == (7, 4)
    assert torch.allclose(weights.sum(-1), torch.ones(7), atol=1e-5)
    assert stats["candidate_ids"].shape == (7, 16)
    assert int(selected.max()) < 128
