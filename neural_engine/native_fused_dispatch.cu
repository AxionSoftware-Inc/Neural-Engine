#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ float gelu(float value) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    return 0.5f * value * (1.0f + erff(value * kInvSqrt2));
}

__global__ void factorized_dispatch_kernel(
    const float* __restrict__ state,
    const int64_t* __restrict__ circuit_ids,
    const float* __restrict__ weights,
    const float* __restrict__ down_factors,
    const float* __restrict__ up_factors,
    const float* __restrict__ bias_factors,
    const float* __restrict__ factor_mix,
    const int64_t* __restrict__ address_factor_ids,
    float* __restrict__ output,
    int64_t tokens,
    int64_t active,
    int64_t factor_count,
    int64_t state_dim,
    int64_t rank) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x);
    const int64_t pair_count = tokens * active;
    if (pair >= pair_count) return;
    const int64_t token = pair / active;
    const int64_t slot = pair - token * active;
    const int64_t address = circuit_ids[pair];
    const int64_t first = address_factor_ids[address * 2];
    const int64_t second = address_factor_ids[address * 2 + 1];
    const float first_mix = factor_mix[address * 2];
    const float second_mix = factor_mix[address * 2 + 1];
    const float* state_row = state + token * state_dim;

    extern __shared__ float hidden[];
    for (int64_t r = threadIdx.x; r < rank; r += blockDim.x) {
        float value = 0.0f;
        for (int64_t d = 0; d < state_dim; ++d) {
            const int64_t first_offset = (first * state_dim + d) * rank + r;
            const int64_t second_offset = (factor_count + second) * state_dim * rank
                                          + d * rank + r;
            value += state_row[d] * (
                first_mix * down_factors[first_offset]
                + second_mix * down_factors[second_offset]);
        }
        hidden[r] = gelu(value);
    }
    __syncthreads();

    const float route = weights[pair];
    float* output_row = output + token * state_dim;
    for (int64_t d = threadIdx.x; d < state_dim; d += blockDim.x) {
        const int64_t first_bias = first * state_dim + d;
        const int64_t second_bias = (factor_count + second) * state_dim + d;
        float value = first_mix * bias_factors[first_bias]
                    + second_mix * bias_factors[second_bias];
        for (int64_t r = 0; r < rank; ++r) {
            const int64_t first_up = (first * rank + r) * state_dim + d;
            const int64_t second_up = ((factor_count + second) * rank + r) * state_dim + d;
            value += hidden[r] * (
                first_mix * up_factors[first_up]
                + second_mix * up_factors[second_up]);
        }
        atomicAdd(output_row + d, route * value);
    }
}

}  // namespace

torch::Tensor native_factorized_dispatch_cuda(
    torch::Tensor state,
    torch::Tensor circuit_ids,
    torch::Tensor weights,
    torch::Tensor down_factors,
    torch::Tensor up_factors,
    torch::Tensor bias_factors,
    torch::Tensor factor_mix,
    torch::Tensor address_factor_ids) {
    TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be float32");
    TORCH_CHECK(circuit_ids.scalar_type() == torch::kInt64, "circuit IDs must be int64");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    TORCH_CHECK(down_factors.scalar_type() == torch::kFloat32, "down factors must be float32");
    TORCH_CHECK(up_factors.scalar_type() == torch::kFloat32, "up factors must be float32");
    TORCH_CHECK(bias_factors.scalar_type() == torch::kFloat32, "bias factors must be float32");
    TORCH_CHECK(factor_mix.scalar_type() == torch::kFloat32, "factor mix must be float32");
    TORCH_CHECK(address_factor_ids.scalar_type() == torch::kInt64,
                "address factor IDs must be int64");
    TORCH_CHECK(state.dim() == 2, "state must be [tokens, state]");
    TORCH_CHECK(circuit_ids.dim() == 2, "circuit IDs must be [tokens, active]");
    TORCH_CHECK(weights.sizes() == circuit_ids.sizes(), "weights shape mismatch");
    TORCH_CHECK(down_factors.dim() == 4 && down_factors.size(0) == 2,
                "down factors must be [2, factors, state, rank]");
    TORCH_CHECK(up_factors.dim() == 4 && up_factors.size(0) == 2,
                "up factors must be [2, factors, rank, state]");
    TORCH_CHECK(bias_factors.dim() == 3 && bias_factors.size(0) == 2,
                "bias factors must be [2, factors, state]");
    TORCH_CHECK(factor_mix.dim() == 2 && factor_mix.size(1) == 2,
                "factor mix must be [addresses, 2]");
    TORCH_CHECK(address_factor_ids.sizes() == factor_mix.sizes(),
                "address factor map shape mismatch");
    const int64_t tokens = state.size(0);
    const int64_t state_dim = state.size(1);
    const int64_t active = circuit_ids.size(1);
    const int64_t factor_count = down_factors.size(1);
    const int64_t rank = down_factors.size(3);
    TORCH_CHECK(circuit_ids.size(0) == tokens, "token dimension mismatch");
    TORCH_CHECK(down_factors.size(2) == state_dim, "down state dimension mismatch");
    TORCH_CHECK(up_factors.size(1) == factor_count && up_factors.size(2) == rank
                && up_factors.size(3) == state_dim, "up factor shape mismatch");
    TORCH_CHECK(bias_factors.size(1) == factor_count && bias_factors.size(2) == state_dim,
                "bias factor shape mismatch");
    TORCH_CHECK(rank <= 1024, "rank exceeds thread/shared-memory safety bound");

    auto output = torch::zeros({tokens, state_dim}, state.options());
    if (tokens == 0 || active == 0) return output;
    const auto stream = at::cuda::getCurrentCUDAStream();
    factorized_dispatch_kernel<<<
        static_cast<unsigned int>(tokens * active), 256,
        static_cast<size_t>(rank) * sizeof(float), stream>>>(
        state.data_ptr<float>(), circuit_ids.data_ptr<int64_t>(),
        weights.data_ptr<float>(), down_factors.data_ptr<float>(),
        up_factors.data_ptr<float>(), bias_factors.data_ptr<float>(),
        factor_mix.data_ptr<float>(), address_factor_ids.data_ptr<int64_t>(),
        output.data_ptr<float>(), tokens, active, factor_count, state_dim, rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
