import torch

from neural_engine.mesoscopic_macro import MesoscopicMacroCellBank


def test_mesoscopic_macro_canonical_parameter_count_and_forward():
    torch.manual_seed(17)
    bank = MesoscopicMacroCellBank(64, 384, 480, 128)
    assert bank.parameters_per_cell == 897504
    assert bank.total_body_parameters == 64 * 897504
    state = torch.randn(3, 384)
    memory = torch.randn(3, 384)
    ids = torch.tensor([[0, 1], [2, 3], [4, -1]])
    weights = torch.tensor([[0.5, 0.5], [0.2, 0.8], [1.0, 0.0]])
    next_state, next_memory = bank(state, memory, ids, weights)
    assert next_state.shape == state.shape
    assert next_memory.shape == memory.shape
    assert torch.isfinite(next_state).all()
    assert torch.isfinite(next_memory).all()
