#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> qwen_router_dispatch_cuda(
    torch::Tensor hidden,
    torch::Tensor first_weight,
    torch::Tensor first_bias,
    torch::Tensor second_weight,
    torch::Tensor second_bias,
    int64_t active_experts,
    double temperature);

std::tuple<torch::Tensor, torch::Tensor> qwen_subset_router_dispatch_cuda(
    torch::Tensor hidden,
    torch::Tensor first_weight,
    torch::Tensor first_bias,
    torch::Tensor second_weight,
    torch::Tensor second_bias,
    torch::Tensor subset_membership,
    int64_t active_experts);

std::tuple<torch::Tensor, torch::Tensor> forward(
    torch::Tensor hidden,
    torch::Tensor first_weight,
    torch::Tensor first_bias,
    torch::Tensor second_weight,
    torch::Tensor second_bias,
    int64_t active_experts,
    double temperature) {
    return qwen_router_dispatch_cuda(
        hidden, first_weight, first_bias, second_weight, second_bias,
        active_experts, temperature
    );
}

std::tuple<torch::Tensor, torch::Tensor> forward_subset(
    torch::Tensor hidden,
    torch::Tensor first_weight,
    torch::Tensor first_bias,
    torch::Tensor second_weight,
    torch::Tensor second_bias,
    torch::Tensor subset_membership,
    int64_t active_experts) {
    return qwen_subset_router_dispatch_cuda(
        hidden, first_weight, first_bias, second_weight, second_bias,
        subset_membership, active_experts
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused Qwen router forward (CUDA)");
    m.def("forward_subset", &forward_subset, "Fused Qwen subset router forward (CUDA)");
}
