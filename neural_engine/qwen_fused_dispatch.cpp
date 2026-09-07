#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> qwen_fused_dispatch_cuda(
    torch::Tensor hidden,
    torch::Tensor top_ids,
    torch::Tensor route_weights,
    torch::Tensor gate_weight,
    torch::Tensor value_weight,
    torch::Tensor output_weight,
    double hard_route_scale);

std::tuple<torch::Tensor, torch::Tensor> forward(
    torch::Tensor hidden,
    torch::Tensor top_ids,
    torch::Tensor route_weights,
    torch::Tensor gate_weight,
    torch::Tensor value_weight,
    torch::Tensor output_weight,
    double hard_route_scale) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
    TORCH_CHECK(top_ids.is_cuda(), "top_ids must be CUDA");
    TORCH_CHECK(route_weights.is_cuda(), "route_weights must be CUDA");
    TORCH_CHECK(gate_weight.is_cuda(), "gate_weight must be CUDA");
    TORCH_CHECK(value_weight.is_cuda(), "value_weight must be CUDA");
    TORCH_CHECK(output_weight.is_cuda(), "output_weight must be CUDA");
    return qwen_fused_dispatch_cuda(
        hidden, top_ids, route_weights, gate_weight, value_weight,
        output_weight, hard_route_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused selected-group Qwen SwiGLU dispatch");
}

