#include <torch/extension.h>

#include <tuple>

std::tuple<torch::Tensor, torch::Tensor> qwen_atomic_pack_cuda(
    torch::Tensor hidden,
    torch::Tensor top_ids,
    int64_t num_experts);

std::tuple<torch::Tensor, torch::Tensor> forward(
    torch::Tensor hidden,
    torch::Tensor top_ids,
    int64_t num_experts) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
    TORCH_CHECK(top_ids.is_cuda(), "top ids must be CUDA");
    return qwen_atomic_pack_cuda(hidden, top_ids, num_experts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Atomic grouped Qwen route packing");
}
