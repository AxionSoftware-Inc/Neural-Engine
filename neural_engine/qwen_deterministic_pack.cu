#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

namespace {

// One block owns one selected (token, slot) pair.  Top-k indices are unique
// per token, so every selected pair has a unique fixed expert-token slot and
// no expert-local counter or sort is needed.  Unselected slots are not read by
// the grouped path and therefore do not need initialization.
__global__ void deterministic_pack_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ top_ids,
    float* __restrict__ grouped_hidden,
    int64_t tokens,
    int64_t active,
    int64_t experts,
    int64_t hidden_size) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x);
    const int64_t pair_count = tokens * active;
    if (pair >= pair_count) return;
    const int64_t token = pair / active;
    const int64_t expert = top_ids[pair];
    if (expert < 0 || expert >= experts) return;
    const float* source = hidden + token * hidden_size;
    float* target = grouped_hidden + (expert * tokens + token) * hidden_size;
    for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        target[h] = source[h];
    }
}

// Same fixed-layout write, but move four adjacent hidden values per thread.
// Qwen hidden sizes are four-aligned, so the row bases remain 16-byte aligned
// for contiguous float32 tensors.  This is an isolated memory-instruction
// probe; routing, layout, and grouped GEMM shapes are unchanged.
__global__ void deterministic_pack_vec4_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ top_ids,
    float* __restrict__ grouped_hidden,
    int64_t tokens,
    int64_t active,
    int64_t experts,
    int64_t hidden_size) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x);
    const int64_t pair_count = tokens * active;
    if (pair >= pair_count) return;
    const int64_t token = pair / active;
    const int64_t expert = top_ids[pair];
    if (expert < 0 || expert >= experts) return;
    const float4* source = reinterpret_cast<const float4*>(
        hidden + token * hidden_size);
    float4* target = reinterpret_cast<float4*>(
        grouped_hidden + (expert * tokens + token) * hidden_size);
    const int64_t hidden_vec4 = hidden_size / 4;
    for (int64_t h = threadIdx.x; h < hidden_vec4; h += blockDim.x) {
        target[h] = source[h];
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
        static_cast<unsigned int>(tokens * active), 256, 0, stream
    >>>(
        hidden.data_ptr<float>(), top_ids.data_ptr<int64_t>(),
        grouped_hidden.data_ptr<float>(), tokens, active, experts,
        hidden_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grouped_hidden;
}

torch::Tensor qwen_deterministic_pack_vec4_cuda(
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
    TORCH_CHECK(hidden_size % 4 == 0,
                "vectorized deterministic pack requires hidden size divisible by 4");
    auto grouped_hidden = torch::empty(
        {experts * tokens, hidden_size}, hidden.options());
    if (tokens == 0 || active == 0) return grouped_hidden;
    const auto stream = at::cuda::getCurrentCUDAStream();
    deterministic_pack_vec4_kernel<<<
        static_cast<unsigned int>(tokens * active), 256, 0, stream
    >>>(
        hidden.data_ptr<float>(), top_ids.data_ptr<int64_t>(),
        grouped_hidden.data_ptr<float>(), tokens, active, experts,
        hidden_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grouped_hidden;
}
