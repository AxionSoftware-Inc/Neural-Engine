#include <torch/extension.h>

torch::Tensor qwen_tiled_grouped_cuda(
    torch::Tensor grouped_hidden,
    torch::Tensor gate_weight,
    torch::Tensor value_weight,
    torch::Tensor output_weight);

torch::Tensor forward(
    torch::Tensor grouped_hidden,
    torch::Tensor gate_weight,
    torch::Tensor value_weight,
    torch::Tensor output_weight) {
    TORCH_CHECK(grouped_hidden.is_cuda(), "grouped hidden must be CUDA");
    TORCH_CHECK(gate_weight.is_cuda(), "gate weights must be CUDA");
    TORCH_CHECK(value_weight.is_cuda(), "value weights must be CUDA");
    TORCH_CHECK(output_weight.is_cuda(), "output weights must be CUDA");
    return qwen_tiled_grouped_cuda(
        grouped_hidden, gate_weight, value_weight, output_weight);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled grouped Qwen SwiGLU dispatch");
}
