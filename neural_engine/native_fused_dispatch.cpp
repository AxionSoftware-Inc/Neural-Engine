#include <torch/extension.h>

torch::Tensor native_factorized_dispatch_cuda(
    torch::Tensor state,
    torch::Tensor circuit_ids,
    torch::Tensor weights,
    torch::Tensor down_factors,
    torch::Tensor up_factors,
    torch::Tensor bias_factors,
    torch::Tensor factor_mix,
    torch::Tensor address_factor_ids);

torch::Tensor forward(
    torch::Tensor state,
    torch::Tensor circuit_ids,
    torch::Tensor weights,
    torch::Tensor down_factors,
    torch::Tensor up_factors,
    torch::Tensor bias_factors,
    torch::Tensor factor_mix,
    torch::Tensor address_factor_ids) {
    TORCH_CHECK(state.is_cuda(), "state must be CUDA");
    TORCH_CHECK(circuit_ids.is_cuda(), "circuit IDs must be CUDA");
    TORCH_CHECK(weights.is_cuda(), "weights must be CUDA");
    TORCH_CHECK(down_factors.is_cuda(), "down factors must be CUDA");
    TORCH_CHECK(up_factors.is_cuda(), "up factors must be CUDA");
    TORCH_CHECK(bias_factors.is_cuda(), "bias factors must be CUDA");
    TORCH_CHECK(factor_mix.is_cuda(), "factor mix must be CUDA");
    TORCH_CHECK(address_factor_ids.is_cuda(), "address factor IDs must be CUDA");
    return native_factorized_dispatch_cuda(
        state, circuit_ids, weights, down_factors, up_factors,
        bias_factors, factor_mix, address_factor_ids);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Native factorized additive circuit dispatch");
}
