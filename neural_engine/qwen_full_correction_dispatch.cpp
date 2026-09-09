#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> qwen_full_correction_dispatch_cuda(
    torch::Tensor hidden,
    torch::Tensor selected_ids,
    torch::Tensor route_weights,
    torch::Tensor gate_weight,
    torch::Tensor value_weight,
    torch::Tensor output_weight,
    torch::Tensor mix_in,
    torch::Tensor mix_out,
    double hard_route_scale);

std::tuple<torch::Tensor, torch::Tensor> forward(
    torch::Tensor hidden,
    torch::Tensor selected_ids,
    torch::Tensor route_weights,
    torch::Tensor gate_weight,
    torch::Tensor value_weight,
    torch::Tensor output_weight,
    torch::Tensor mix_in,
    torch::Tensor mix_out,
    double hard_route_scale) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
    TORCH_CHECK(selected_ids.is_cuda(), "selected ids must be CUDA");
    TORCH_CHECK(route_weights.is_cuda(), "route weights must be CUDA");
    TORCH_CHECK(gate_weight.is_cuda(), "gate weights must be CUDA");
    TORCH_CHECK(value_weight.is_cuda(), "value weights must be CUDA");
    TORCH_CHECK(output_weight.is_cuda(), "output weights must be CUDA");
    TORCH_CHECK(mix_in.is_cuda(), "mix_in must be CUDA");
    TORCH_CHECK(mix_out.is_cuda(), "mix_out must be CUDA");
    return qwen_full_correction_dispatch_cuda(
        hidden, selected_ids, route_weights, gate_weight, value_weight,
        output_weight, mix_in, mix_out, hard_route_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused selected Qwen output and correction");
}
