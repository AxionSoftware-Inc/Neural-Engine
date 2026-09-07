import torch

from neural_engine.p001_sparse_output_selector import SparseOutputSignatureSelector


def test_bank_initialization_matches_projected_source_circuit():
    torch.manual_seed(17)
    state_dim = 6
    source_rank = 3
    signature_dim = 4
    selector = SparseOutputSignatureSelector(
        state_dim=state_dim,
        num_circuits=5,
        candidate_pool=3,
        active_circuits=2,
        signature_rank=source_rank,
        signature_dim=signature_dim,
    )
    down = torch.randn(5, state_dim, source_rank)
    up = torch.randn(5, source_rank, state_dim)
    bias = torch.randn(5, state_dim)
    projection = torch.randn(state_dim, signature_dim)
    selector.initialize_from_circuit_bank(down, up, bias, projection)

    query = torch.randn(2, state_dim)
    candidate_ids = torch.tensor([[0, 2, 4], [1, 3, 4]])
    actual_hidden = torch.einsum("bd,bmdr->bmr", query, down[candidate_ids])
    actual_hidden = torch.nn.functional.gelu(actual_hidden)
    actual = torch.einsum("bmr,bmrd->bmd", actual_hidden, up[candidate_ids])
    actual = actual + bias[candidate_ids]
    expected = torch.einsum("bmd,ds->bms", actual, projection)

    torch.testing.assert_close(
        selector.candidate_signatures(query, candidate_ids), expected
    )
