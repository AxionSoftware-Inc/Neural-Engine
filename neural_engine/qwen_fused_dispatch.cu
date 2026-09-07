#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

// One block owns one selected (token, group) pair.  The coefficient vector is
// computed once in shared memory, then the block writes the selected output
// and atomically accumulates its weighted contribution.  This avoids the
// Python-side expert loop and launches one fused kernel for the selected path.
__global__ void dispatch_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ top_ids,
    const float* __restrict__ route_weights,
    const float* __restrict__ gate_weight,
    const float* __restrict__ value_weight,
    const float* __restrict__ output_weight,
    float* __restrict__ selected,
    float* __restrict__ output,
    int64_t tokens,
    int64_t active,
    int64_t experts,
    int64_t hidden_size,
    int64_t group_size,
    float scale) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x);
    const int64_t pair_count = tokens * active;
    if (pair >= pair_count) return;
    const int64_t token = pair / active;
    const int64_t slot = pair - token * active;
    const int64_t expert = top_ids[pair];
    if (expert < 0 || expert >= experts) return;

    extern __shared__ float coefficient[];
    const float* hidden_row = hidden + token * hidden_size;
    const float* gate_row = gate_weight + expert * group_size * hidden_size;
    const float* value_row = value_weight + expert * group_size * hidden_size;
    for (int64_t c = threadIdx.x; c < group_size; c += blockDim.x) {
        float gate = 0.0f;
        float value = 0.0f;
        const int64_t offset = c * hidden_size;
        for (int64_t h = 0; h < hidden_size; ++h) {
            const float input = hidden_row[h];
            gate += input * gate_row[offset + h];
            value += input * value_row[offset + h];
        }
        coefficient[c] = silu(gate) * value;
    }
    __syncthreads();

    const float route = route_weights[pair];
    float* selected_row = selected + pair * hidden_size;
    float* output_row = output + token * hidden_size;
    const float* down_row = output_weight + expert * hidden_size * group_size;
    for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float value = 0.0f;
        const int64_t offset = h * group_size;
        for (int64_t c = 0; c < group_size; ++c) {
            value += coefficient[c] * down_row[offset + c];
        }
        selected_row[h] = value;
        atomicAdd(output_row + h, scale * route * value);
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> qwen_fused_dispatch_cuda(
    torch::Tensor hidden,
    torch::Tensor top_ids,
    torch::Tensor route_weights,
    torch::Tensor gate_weight,
    torch::Tensor value_weight,
    torch::Tensor output_weight,
    double hard_route_scale) {
    TORCH_CHECK(hidden.scalar_type() == torch::kFloat32, "hidden must be float32");
    TORCH_CHECK(route_weights.scalar_type() == torch::kFloat32, "route weights must be float32");
    TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat32, "gate weights must be float32");
    TORCH_CHECK(value_weight.scalar_type() == torch::kFloat32, "value weights must be float32");
    TORCH_CHECK(output_weight.scalar_type() == torch::kFloat32, "output weights must be float32");
    TORCH_CHECK(top_ids.scalar_type() == torch::kInt64, "top ids must be int64");
    TORCH_CHECK(hidden.dim() == 2, "hidden must be [tokens, hidden]");
    TORCH_CHECK(top_ids.dim() == 2, "top ids must be [tokens, active]");
    TORCH_CHECK(route_weights.sizes() == top_ids.sizes(), "route weights shape mismatch");
    TORCH_CHECK(gate_weight.dim() == 3, "gate weights must be [experts, group, hidden]");
    TORCH_CHECK(value_weight.sizes() == gate_weight.sizes(), "value weights shape mismatch");
    TORCH_CHECK(output_weight.dim() == 3, "output weights must be [experts, hidden, group]");
    const int64_t tokens = hidden.size(0);
    const int64_t hidden_size = hidden.size(1);
    const int64_t experts = gate_weight.size(0);
    const int64_t group_size = gate_weight.size(1);
    const int64_t active = top_ids.size(1);
    TORCH_CHECK(top_ids.size(0) == tokens, "top ids token dimension mismatch");
    TORCH_CHECK(gate_weight.size(2) == hidden_size, "gate hidden dimension mismatch");
    TORCH_CHECK(output_weight.size(0) == experts, "output expert dimension mismatch");
    TORCH_CHECK(output_weight.size(1) == hidden_size, "output hidden dimension mismatch");
    TORCH_CHECK(output_weight.size(2) == group_size, "output group dimension mismatch");
    TORCH_CHECK(group_size <= 49152, "group size exceeds shared-memory safety bound");

    auto selected = torch::empty({tokens, active, hidden_size}, hidden.options());
    auto output = torch::zeros({tokens, hidden_size}, hidden.options());
    if (tokens == 0 || active == 0) {
        return std::make_tuple(selected, output);
    }
    const int threads = 256;
    const auto blocks = static_cast<unsigned int>(tokens * active);
    const auto stream = at::cuda::getDefaultCUDAStream();
    dispatch_kernel<<<blocks, threads, group_size * sizeof(float), stream>>>(
        hidden.data_ptr<float>(), top_ids.data_ptr<int64_t>(),
        route_weights.data_ptr<float>(), gate_weight.data_ptr<float>(),
        value_weight.data_ptr<float>(), output_weight.data_ptr<float>(),
        selected.data_ptr<float>(), output.data_ptr<float>(), tokens, active,
        experts, hidden_size, group_size, static_cast<float>(hard_route_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(selected, output);
}

