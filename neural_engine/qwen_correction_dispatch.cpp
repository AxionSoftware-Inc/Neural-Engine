#include <torch/extension.h>

#include <tuple>

torch::Tensor qwen_correction_dispatch_cuda(
    torch::Tensor selected_outputs,
    torch::Tensor selected_ids,
    torch::Tensor route_weights,
    torch::Tensor mix_in,
    torch::Tensor mix_out,
    double hard_route_scale);

torch::Tensor forward(
    torch::Tensor selected_outputs,
    torch::Tensor selected_ids,
    torch::Tensor route_weights,
    torch::Tensor mix_in,
    torch::Tensor mix_out,
    double hard_route_scale) {
    TORCH_CHECK(selected_outputs.is_cuda(), "selected outputs must be CUDA");
    TORCH_CHECK(selected_ids.is_cuda(), "selected ids must be CUDA");
    TORCH_CHECK(route_weights.is_cuda(), "route weights must be CUDA");
    TORCH_CHECK(mix_in.is_cuda(), "mix_in must be CUDA");
    TORCH_CHECK(mix_out.is_cuda(), "mix_out must be CUDA");
    TORCH_CHECK(selected_outputs.is_contiguous(), "selected outputs must be contiguous");
    TORCH_CHECK(selected_ids.is_contiguous(), "selected ids must be contiguous");
    TORCH_CHECK(route_weights.is_contiguous(), "route weights must be contiguous");
    TORCH_CHECK(mix_in.is_contiguous(), "mix_in must be contiguous");
    TORCH_CHECK(mix_out.is_contiguous(), "mix_out must be contiguous");
    return qwen_correction_dispatch_cuda(
        selected_outputs, selected_ids, route_weights, mix_in, mix_out,
        hard_route_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Fused selected correction dispatch");
}
