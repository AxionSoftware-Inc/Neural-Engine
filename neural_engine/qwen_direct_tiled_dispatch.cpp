#include <torch/extension.h>

torch::Tensor qwen_direct_tiled_cuda(
    torch::Tensor flat_hidden,
    torch::Tensor sorted_token_ids,
    torch::Tensor sorted_slots,
    torch::Tensor starts,
    torch::Tensor counts,
    torch::Tensor route_weights,
    torch::Tensor group_gate_weight,
    torch::Tensor group_value_weight,
    torch::Tensor group_output_weight,
    int64_t active_experts,
    double hard_route_scale,
    bool uniform_accum);

torch::Tensor forward(
    torch::Tensor flat_hidden,
    torch::Tensor sorted_token_ids,
    torch::Tensor sorted_slots,
    torch::Tensor starts,
    torch::Tensor counts,
    torch::Tensor route_weights,
    torch::Tensor group_gate_weight,
    torch::Tensor group_value_weight,
    torch::Tensor group_output_weight,
    int64_t active_experts,
    double hard_route_scale,
    bool uniform_accum) {
    TORCH_CHECK(flat_hidden.is_cuda(), "hidden must be CUDA");
    TORCH_CHECK(sorted_token_ids.is_cuda(), "token ids must be CUDA");
    TORCH_CHECK(sorted_slots.is_cuda(), "slots must be CUDA");
    TORCH_CHECK(starts.is_cuda(), "starts must be CUDA");
    TORCH_CHECK(counts.is_cuda(), "counts must be CUDA");
    TORCH_CHECK(route_weights.is_cuda(), "route weights must be CUDA");
    TORCH_CHECK(group_gate_weight.is_cuda(), "gate weights must be CUDA");
    TORCH_CHECK(group_value_weight.is_cuda(), "value weights must be CUDA");
    TORCH_CHECK(group_output_weight.is_cuda(), "output weights must be CUDA");
    return qwen_direct_tiled_cuda(
        flat_hidden, sorted_token_ids, sorted_slots, starts, counts,
        route_weights, group_gate_weight, group_value_weight,
        group_output_weight, active_experts, hard_route_scale, uniform_accum);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Direct tiled Qwen grouped dispatch");
}
