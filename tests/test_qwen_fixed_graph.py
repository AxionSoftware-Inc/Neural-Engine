import pytest
import torch

from neural_engine.qwen_fixed_graph import greedy_generate_fixed_shape


def test_fixed_graph_zero_token_request_is_identity_on_cpu():
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    result = greedy_generate_fixed_shape(
        object(), input_ids, 0, use_cuda_graph=False,
    )
    assert torch.equal(result, input_ids)
    assert result.data_ptr() != input_ids.data_ptr()


def test_fixed_graph_rejects_non_cuda_input():
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with pytest.raises(ValueError, match="requires CUDA"):
        greedy_generate_fixed_shape(
            object(), input_ids, 1, use_cuda_graph=True,
        )
