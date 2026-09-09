#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

namespace {

// One block owns one [expert, token] output row.  A selected route is copied
// to its deterministic expert-token slot; unselected slots are zeroed.  No
// expert-local counter or sort is needed, and the caller can map a pair to
// expert*tokens + token directly.
__global__ void deterministic_pack_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ top_ids,
    float* __restrict__ grouped_hidden,
    int64_t tokens,
    int64_t active,
    int64_t experts,
    int64_t hidden_size) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int64_t total_rows = experts * tokens;
    if (row >= total_rows) return;
    const int64_t expert = row / tokens;
    const int64_t token = row - expert * tokens;
    __shared__ int64_t selected;
    if (threadIdx.x == 0) {
        selected = -1;
        for (int64_t slot = 0; slot < active; ++slot) {
            if (top_ids[token * active + slot] == expert) {
                selected = slot;
                break;
            }
        }
    }
    __syncthreads();
    const float* source = hidden + token * hidden_size;
    float* target = grouped_hidden + row * hidden_size;
    if (selected >= 0) {
        for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            target[h] = source[h];
        }
    } else {
        for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            target[h] = 0.0f;
        }
    }
}

}  // namespace

torch::Tensor qwen_deterministic_pack_cuda(
    torch::Tensor hidden,
    torch::Tensor top_ids,
    int64_t experts) {
    TORCH_CHECK(hidden.scalar_type() == torch::kFloat32,
                "hidden must be float32");
    TORCH_CHECK(top_ids.scalar_type() == torch::kInt64,
                "top ids must be int64");
    TORCH_CHECK(hidden.dim() == 2 && top_ids.dim() == 2,
                "hidden/top ids must be rank-2");
    TORCH_CHECK(hidden.is_contiguous() && top_ids.is_contiguous(),
                "deterministic pack inputs must be contiguous");
    const int64_t tokens = hidden.size(0);
    const int64_t hidden_size = hidden.size(1);
    const int64_t active = top_ids.size(1);
    TORCH_CHECK(top_ids.size(0) == tokens, "token dimension mismatch");
    TORCH_CHECK(experts > 0, "num_experts must be positive");
    auto grouped_hidden = torch::empty(
        {experts * tokens, hidden_size}, hidden.options());
    if (tokens == 0 || active == 0) return grouped_hidden;
    const auto stream = at::cuda::getCurrentCUDAStream();
    deterministic_pack_kernel<<<
        static_cast<unsigned int>(experts * tokens), 256, 0, stream
    >>>(
        hidden.data_ptr<float>(), top_ids.data_ptr<int64_t>(),
        grouped_hidden.data_ptr<float>(), tokens, active, experts,
        hidden_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grouped_hidden;
}
