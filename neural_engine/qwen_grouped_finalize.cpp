#include <torch/extension.h>

torch::Tensor qwen_grouped_finalize_cuda(
    torch::Tensor grouped_output,
    torch::Tensor packed_positions,
    torch::Tensor token_ids,
    torch::Tensor slots,
    torch::Tensor route_weights,
    double hard_route_scale);

torch::Tensor qwen_grouped_finalize_token_cuda(
    torch::Tensor grouped_output,
    torch::Tensor packed_positions,
    torch::Tensor route_weights,
    double hard_route_scale);

torch::Tensor forward(
    torch::Tensor grouped_output,
    torch::Tensor packed_positions,
    torch::Tensor token_ids,
    torch::Tensor slots,
    torch::Tensor route_weights,
    double hard_route_scale) {
    TORCH_CHECK(grouped_output.is_cuda(), "grouped output must be CUDA");
    TORCH_CHECK(packed_positions.is_cuda(), "packed positions must be CUDA");
    TORCH_CHECK(token_ids.is_cuda(), "token ids must be CUDA");
    TORCH_CHECK(slots.is_cuda(), "slots must be CUDA");
    TORCH_CHECK(route_weights.is_cuda(), "route weights must be CUDA");
    return qwen_grouped_finalize_cuda(
        grouped_output, packed_positions, token_ids, slots, route_weights,
        hard_route_scale);
}

torch::Tensor forward_token(
    torch::Tensor grouped_output,
    torch::Tensor packed_positions,
    torch::Tensor route_weights,
    double hard_route_scale) {
    TORCH_CHECK(grouped_output.is_cuda(), "grouped output must be CUDA");
    TORCH_CHECK(packed_positions.is_cuda(), "packed positions must be CUDA");
    TORCH_CHECK(route_weights.is_cuda(), "route weights must be CUDA");
    return qwen_grouped_finalize_token_cuda(
        grouped_output, packed_positions, route_weights, hard_route_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Finalize grouped Qwen output");
    m.def("forward_token", &forward_token,
          "Finalize grouped Qwen output with token-owned reduction");
}
