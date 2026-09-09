#include <torch/extension.h>

torch::Tensor qwen_deterministic_pack_cuda(
    torch::Tensor hidden,
    torch::Tensor top_ids,
    int64_t experts);

torch::Tensor forward(
    torch::Tensor hidden,
    torch::Tensor top_ids,
    int64_t experts) {
    TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
    TORCH_CHECK(top_ids.is_cuda(), "top ids must be CUDA");
    return qwen_deterministic_pack_cuda(hidden, top_ids, experts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Deterministic fixed-layout Qwen route pack");
}
