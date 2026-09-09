#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

namespace {

// One block owns one selected pair.  Grouped projection remains a batched
// matmul; this kernel only fuses the subsequent gather and token reduction.
// It is used after folded correction, where no selected-output tensor is
// needed by the wrapper.
__global__ void finalize_kernel(
    const float* __restrict__ grouped_output,
    const int64_t* __restrict__ packed_positions,
    const int64_t* __restrict__ token_ids,
    const int64_t* __restrict__ slots,
    const float* __restrict__ route_weights,
    float* __restrict__ output,
    int64_t pairs,
    int64_t active,
    int64_t hidden_size,
    float hard_route_scale) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x);
    if (pair >= pairs) return;
    const int64_t token = token_ids[pair];
    const int64_t slot = slots[pair];
    const float scale = hard_route_scale * route_weights[token * active + slot];
    const float* source = grouped_output + packed_positions[pair] * hidden_size;
    float* target = output + token * hidden_size;
    for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        atomicAdd(target + h, scale * source[h]);
    }
}

// One block owns one token and reduces all K selected expert outputs in a
// fixed slot order. This avoids K competing atomicAdd writers per token.
__global__ void token_finalize_kernel(
    const float* __restrict__ grouped_output,
    const int64_t* __restrict__ packed_positions,
    const float* __restrict__ route_weights,
    float* __restrict__ output,
    int64_t tokens,
    int64_t active,
    int64_t hidden_size,
    float hard_route_scale) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= tokens) return;
    float* target = output + token * hidden_size;
    for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float sum = 0.0f;
        for (int64_t slot = 0; slot < active; ++slot) {
            const int64_t pair = token * active + slot;
            const float scale =
                hard_route_scale * route_weights[pair];
            const float* source =
                grouped_output + packed_positions[pair] * hidden_size;
            sum += scale * source[h];
        }
        target[h] = sum;
    }
}

// Fixed-layout variant: derive the grouped row directly from the original
// [token, active] expert IDs.  This removes the int64 packed-position buffer
// and its construction from the fixed-pack path.
__global__ void token_finalize_ids_kernel(
    const float* __restrict__ grouped_output,
    const int64_t* __restrict__ top_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ output,
    int64_t tokens,
    int64_t active,
    int64_t experts,
    int64_t hidden_size,
    float hard_route_scale) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= tokens) return;
    float* target = output + token * hidden_size;
    for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float sum = 0.0f;
        for (int64_t slot = 0; slot < active; ++slot) {
            const int64_t pair = token * active + slot;
            const int64_t expert = top_ids[pair];
            if (expert < 0 || expert >= experts) continue;
            const float scale =
                hard_route_scale * route_weights[pair];
            const float* source = grouped_output +
                (expert * tokens + token) * hidden_size;
            sum += scale * source[h];
        }
        target[h] = sum;
    }
}

}  // namespace

torch::Tensor qwen_grouped_finalize_cuda(
    torch::Tensor grouped_output,
    torch::Tensor packed_positions,
    torch::Tensor token_ids,
    torch::Tensor slots,
    torch::Tensor route_weights,
    double hard_route_scale) {
    TORCH_CHECK(grouped_output.scalar_type() == torch::kFloat32,
                "grouped output must be float32");
    TORCH_CHECK(packed_positions.scalar_type() == torch::kInt64,
                "packed positions must be int64");
    TORCH_CHECK(token_ids.scalar_type() == torch::kInt64,
                "token ids must be int64");
    TORCH_CHECK(slots.scalar_type() == torch::kInt64,
                "slots must be int64");
    TORCH_CHECK(route_weights.scalar_type() == torch::kFloat32,
                "route weights must be float32");
    TORCH_CHECK(grouped_output.dim() == 2,
                "grouped output must be [packed rows, hidden]");
    TORCH_CHECK(packed_positions.dim() == 1 && token_ids.dim() == 1 &&
                    slots.dim() == 1,
                "position metadata must be rank-1");
    TORCH_CHECK(route_weights.dim() == 2,
                "route weights must be [tokens, active]");
    const int64_t pairs = packed_positions.size(0);
    const int64_t tokens = route_weights.size(0);
    const int64_t active = route_weights.size(1);
    const int64_t hidden_size = grouped_output.size(1);
    TORCH_CHECK(token_ids.size(0) == pairs && slots.size(0) == pairs,
                "pair metadata size mismatch");
    TORCH_CHECK(grouped_output.is_contiguous() &&
                    packed_positions.is_contiguous() &&
                    token_ids.is_contiguous() && slots.is_contiguous() &&
                    route_weights.is_contiguous(),
                "grouped finalize inputs must be contiguous");

    auto output = torch::zeros({tokens, hidden_size}, grouped_output.options());
    if (pairs == 0 || tokens == 0 || hidden_size == 0) {
        return output;
    }
    const auto stream = at::cuda::getCurrentCUDAStream();
    finalize_kernel<<<static_cast<unsigned int>(pairs), 256, 0, stream>>>(
        grouped_output.data_ptr<float>(), packed_positions.data_ptr<int64_t>(),
        token_ids.data_ptr<int64_t>(), slots.data_ptr<int64_t>(),
        route_weights.data_ptr<float>(), output.data_ptr<float>(), pairs,
        active, hidden_size, static_cast<float>(hard_route_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor qwen_grouped_finalize_token_cuda(
    torch::Tensor grouped_output,
    torch::Tensor packed_positions,
    torch::Tensor route_weights,
    double hard_route_scale) {
    TORCH_CHECK(grouped_output.scalar_type() == torch::kFloat32,
                "grouped output must be float32");
    TORCH_CHECK(packed_positions.scalar_type() == torch::kInt64,
                "packed positions must be int64");
    TORCH_CHECK(route_weights.scalar_type() == torch::kFloat32,
                "route weights must be float32");
    TORCH_CHECK(grouped_output.dim() == 2,
                "grouped output must be [packed rows, hidden]");
    TORCH_CHECK(packed_positions.dim() == 1,
                "packed positions must be rank-1");
    TORCH_CHECK(route_weights.dim() == 2,
                "route weights must be [tokens, active]");
    const int64_t tokens = route_weights.size(0);
    const int64_t active = route_weights.size(1);
    const int64_t hidden_size = grouped_output.size(1);
    TORCH_CHECK(packed_positions.numel() == tokens * active,
                "packed positions size mismatch");
    TORCH_CHECK(grouped_output.is_contiguous() &&
                    packed_positions.is_contiguous() &&
                    route_weights.is_contiguous(),
                "grouped token finalize inputs must be contiguous");

    auto output = torch::empty({tokens, hidden_size}, grouped_output.options());
    if (tokens == 0 || active == 0 || hidden_size == 0) {
        return output.zero_();
    }
    const auto stream = at::cuda::getCurrentCUDAStream();
    token_finalize_kernel<<<static_cast<unsigned int>(tokens), 256, 0, stream>>>(
        grouped_output.data_ptr<float>(), packed_positions.data_ptr<int64_t>(),
        route_weights.data_ptr<float>(), output.data_ptr<float>(), tokens,
        active, hidden_size, static_cast<float>(hard_route_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor qwen_grouped_finalize_token_ids_cuda(
    torch::Tensor grouped_output,
    torch::Tensor top_ids,
    torch::Tensor route_weights,
    int64_t experts,
    double hard_route_scale) {
    TORCH_CHECK(grouped_output.scalar_type() == torch::kFloat32,
                "grouped output must be float32");
    TORCH_CHECK(top_ids.scalar_type() == torch::kInt64,
                "top ids must be int64");
    TORCH_CHECK(route_weights.scalar_type() == torch::kFloat32,
                "route weights must be float32");
    TORCH_CHECK(grouped_output.dim() == 2,
                "grouped output must be [packed rows, hidden]");
    TORCH_CHECK(top_ids.dim() == 2,
                "top ids must be [tokens, active]");
    TORCH_CHECK(route_weights.dim() == 2,
                "route weights must be [tokens, active]");
    const int64_t tokens = route_weights.size(0);
    const int64_t active = route_weights.size(1);
    const int64_t hidden_size = grouped_output.size(1);
    TORCH_CHECK(top_ids.size(0) == tokens && top_ids.size(1) == active,
                "top ids/route weights size mismatch");
    TORCH_CHECK(experts > 0, "num experts must be positive");
    TORCH_CHECK(grouped_output.size(0) == experts * tokens,
                "fixed grouped output row count mismatch");
    TORCH_CHECK(grouped_output.is_contiguous() && top_ids.is_contiguous() &&
                    route_weights.is_contiguous(),
                "grouped token-id finalization inputs must be contiguous");

    auto output = torch::empty({tokens, hidden_size}, grouped_output.options());
    if (tokens == 0 || active == 0 || hidden_size == 0) {
        return output.zero_();
    }
    const auto stream = at::cuda::getCurrentCUDAStream();
    token_finalize_ids_kernel<<<static_cast<unsigned int>(tokens), 256, 0, stream>>>(
        grouped_output.data_ptr<float>(), top_ids.data_ptr<int64_t>(),
        route_weights.data_ptr<float>(), output.data_ptr<float>(), tokens,
        active, experts, hidden_size, static_cast<float>(hard_route_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
