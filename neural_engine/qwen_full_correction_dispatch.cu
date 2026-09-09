#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cstdint>
#include <tuple>

namespace {

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

// One block owns one selected (token, group) pair.  It computes the selected
// Qwen SwiGLU output, projects that output through the low-rank correction,
// and accumulates the combined result in one launch.  The selected output is
// still written because the parent route contract exposes it for diagnostics.
__global__ void full_correction_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ selected_ids,
    const float* __restrict__ route_weights,
    const float* __restrict__ gate_weight,
    const float* __restrict__ value_weight,
    const float* __restrict__ output_weight,
    const float* __restrict__ mix_in,
    const float* __restrict__ mix_out,
    float* __restrict__ selected,
    float* __restrict__ output,
    int64_t tokens,
    int64_t active,
    int64_t experts,
    int64_t hidden_size,
    int64_t group_size,
    int64_t rank,
    float scale) {
    const int64_t pair = static_cast<int64_t>(blockIdx.x);
    const int64_t pair_count = tokens * active;
    if (pair >= pair_count) return;
    const int64_t token = pair / active;
    const int64_t expert = selected_ids[pair];
    if (expert < 0 || expert >= experts) return;

    extern __shared__ float shared[];
    float* coefficient = shared;
    float* selected_vector = coefficient + group_size;
    float* latent = selected_vector + hidden_size;
    const int warp_count = (blockDim.x + 31) >> 5;
    float* warp_sums = latent + rank;

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

    const float* down_row = output_weight + expert * hidden_size * group_size;
    float* selected_row = selected + pair * hidden_size;
    for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float value = 0.0f;
        const int64_t offset = h * group_size;
        for (int64_t c = 0; c < group_size; ++c) {
            value += coefficient[c] * down_row[offset + c];
        }
        selected_vector[h] = value;
        selected_row[h] = value;
    }
    __syncthreads();

    const float* mix_in_row = mix_in + expert * rank * hidden_size;
    for (int64_t r = 0; r < rank; ++r) {
        float partial = 0.0f;
        const int64_t offset = r * hidden_size;
        for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
            partial += selected_vector[h] * mix_in_row[offset + h];
        }
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        for (int delta = 16; delta > 0; delta >>= 1) {
            partial += __shfl_down_sync(0xffffffff, partial, delta);
        }
        if (lane == 0) warp_sums[r * warp_count + warp] = partial;
        __syncthreads();
        if (threadIdx.x == 0) {
            float total = 0.0f;
            for (int w = 0; w < warp_count; ++w) {
                total += warp_sums[r * warp_count + w];
            }
            latent[r] = total;
        }
        __syncthreads();
    }

    const float* mix_out_row = mix_out + expert * hidden_size * rank;
    const float route = route_weights[pair];
    float* output_row = output + token * hidden_size;
    for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        float correction = 0.0f;
        const int64_t offset = h * rank;
        for (int64_t r = 0; r < rank; ++r) {
            correction += latent[r] * mix_out_row[offset + r];
        }
        atomicAdd(output_row + h, scale * route *
            (selected_vector[h] + correction));
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> qwen_full_correction_dispatch_cuda(
    torch::Tensor hidden,
    torch::Tensor selected_ids,
    torch::Tensor route_weights,
    torch::Tensor gate_weight,
    torch::Tensor value_weight,
    torch::Tensor output_weight,
    torch::Tensor mix_in,
    torch::Tensor mix_out,
    double hard_route_scale) {
    TORCH_CHECK(hidden.scalar_type() == torch::kFloat32, "hidden must be float32");
    TORCH_CHECK(selected_ids.scalar_type() == torch::kInt64, "selected ids must be int64");
    TORCH_CHECK(route_weights.scalar_type() == torch::kFloat32, "route weights must be float32");
    TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat32, "gate weights must be float32");
    TORCH_CHECK(value_weight.scalar_type() == torch::kFloat32, "value weights must be float32");
    TORCH_CHECK(output_weight.scalar_type() == torch::kFloat32, "output weights must be float32");
    TORCH_CHECK(mix_in.scalar_type() == torch::kFloat32, "mix_in must be float32");
    TORCH_CHECK(mix_out.scalar_type() == torch::kFloat32, "mix_out must be float32");
    TORCH_CHECK(hidden.dim() == 2, "hidden must be [tokens, hidden]");
    TORCH_CHECK(selected_ids.dim() == 2, "selected ids must be [tokens, active]");
    TORCH_CHECK(route_weights.sizes() == selected_ids.sizes(), "route weights shape mismatch");
    TORCH_CHECK(gate_weight.dim() == 3, "gate weights must be [experts, group, hidden]");
    TORCH_CHECK(value_weight.sizes() == gate_weight.sizes(), "value weights shape mismatch");
    TORCH_CHECK(output_weight.dim() == 3, "output weights must be [experts, hidden, group]");
    TORCH_CHECK(mix_in.dim() == 3, "mix_in must be [experts, rank, hidden]");
    TORCH_CHECK(mix_out.dim() == 3, "mix_out must be [experts, hidden, rank]");
    const int64_t tokens = hidden.size(0);
    const int64_t hidden_size = hidden.size(1);
    const int64_t experts = gate_weight.size(0);
    const int64_t group_size = gate_weight.size(1);
    const int64_t active = selected_ids.size(1);
    const int64_t rank = mix_in.size(1);
    TORCH_CHECK(selected_ids.size(0) == tokens, "selected ids token mismatch");
    TORCH_CHECK(gate_weight.size(2) == hidden_size, "gate hidden mismatch");
    TORCH_CHECK(output_weight.size(0) == experts, "output expert mismatch");
    TORCH_CHECK(output_weight.size(1) == hidden_size, "output hidden mismatch");
    TORCH_CHECK(output_weight.size(2) == group_size, "output group mismatch");
    TORCH_CHECK(mix_in.size(0) == experts, "mix_in expert mismatch");
    TORCH_CHECK(mix_in.size(2) == hidden_size, "mix_in hidden mismatch");
    TORCH_CHECK(mix_out.size(0) == experts, "mix_out expert mismatch");
    TORCH_CHECK(mix_out.size(1) == hidden_size, "mix_out hidden mismatch");
    TORCH_CHECK(mix_out.size(2) == rank, "mix_out rank mismatch");

    auto selected = torch::empty({tokens, active, hidden_size}, hidden.options());
    auto output = torch::zeros({tokens, hidden_size}, hidden.options());
    if (tokens == 0 || active == 0 || hidden_size == 0) {
        return std::make_tuple(selected, output);
    }
    const int threads = 256;
    const int warp_count = (threads + 31) / 32;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const size_t shared_bytes = static_cast<size_t>(
        group_size + hidden_size + rank + warp_count * rank) * sizeof(float);
    full_correction_kernel<<<
        static_cast<unsigned int>(tokens * active), threads,
        shared_bytes, stream
    >>>(
        hidden.data_ptr<float>(), selected_ids.data_ptr<int64_t>(),
        route_weights.data_ptr<float>(), gate_weight.data_ptr<float>(),
        value_weight.data_ptr<float>(), output_weight.data_ptr<float>(),
        mix_in.data_ptr<float>(), mix_out.data_ptr<float>(),
        selected.data_ptr<float>(), output.data_ptr<float>(), tokens, active,
        experts, hidden_size, group_size, rank,
        static_cast<float>(hard_route_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(selected, output);
}
