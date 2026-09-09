#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <tuple>

namespace {

// One block owns one selected (token, slot) pair. The block reserves an
// expert-local row with an atomic counter, then copies the hidden row into the
// expert-major workspace. The returned position maps the original pair order
// directly to that workspace, so the caller does not need argsort/scatter.
__global__ void atomic_pack_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ top_ids,
    float* __restrict__ grouped_hidden,
    int64_t* __restrict__ packed_positions,
    int32_t* __restrict__ counters,
    int64_t tokens,
    int64_t active,
    int64_t experts,
    int64_t hidden_size) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x);
    const int64_t pair_count = tokens * active;
    if (pair >= pair_count) return;

    const int64_t expert = top_ids[pair];
    __shared__ int64_t destination;
    if (threadIdx.x == 0) {
        if (expert >= 0 && expert < experts) {
            const int32_t row = atomicAdd(counters + expert, 1);
            destination = expert * tokens + static_cast<int64_t>(row);
            packed_positions[pair] = destination;
        } else {
            destination = -1;
            packed_positions[pair] = -1;
        }
    }
    __syncthreads();
    if (destination < 0) return;

    const int64_t token = pair / active;
    const float* source = hidden + token * hidden_size;
    float* target = grouped_hidden + destination * hidden_size;
    for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        target[h] = source[h];
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> qwen_atomic_pack_cuda(
    torch::Tensor hidden,
    torch::Tensor top_ids,
    int64_t experts) {
    TORCH_CHECK(hidden.scalar_type() == torch::kFloat32,
                "hidden must be float32");
    TORCH_CHECK(top_ids.scalar_type() == torch::kInt64,
                "top ids must be int64");
    TORCH_CHECK(hidden.dim() == 2, "hidden must be [tokens, hidden]");
    TORCH_CHECK(top_ids.dim() == 2, "top ids must be [tokens, active]");
    TORCH_CHECK(hidden.is_contiguous() && top_ids.is_contiguous(),
                "atomic pack inputs must be contiguous");

    const int64_t tokens = hidden.size(0);
    const int64_t hidden_size = hidden.size(1);
    const int64_t active = top_ids.size(1);
    TORCH_CHECK(top_ids.size(0) == tokens, "token dimension mismatch");
    TORCH_CHECK(experts > 0, "num_experts must be positive");

    auto grouped_hidden = torch::empty(
        {experts * tokens, hidden_size}, hidden.options());
    auto packed_positions = torch::empty(
        {tokens * active}, top_ids.options());
    auto counters = torch::zeros(
        {experts}, top_ids.options().dtype(torch::kInt32));
    if (tokens == 0 || active == 0 || experts == 0) {
        return std::make_tuple(grouped_hidden, packed_positions);
    }

    const auto stream = at::cuda::getCurrentCUDAStream();
    atomic_pack_kernel<<<
        static_cast<unsigned int>(tokens * active), 256, 0, stream
    >>>(
        hidden.data_ptr<float>(), top_ids.data_ptr<int64_t>(),
        grouped_hidden.data_ptr<float>(), packed_positions.data_ptr<int64_t>(),
        counters.data_ptr<int32_t>(), tokens, active, experts, hidden_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(grouped_hidden, packed_positions);
}
