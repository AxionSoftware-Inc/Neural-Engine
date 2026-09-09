#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

namespace {

__global__ void correction_kernel(
    const float* __restrict__ selected_outputs,
    const int64_t* __restrict__ selected_ids,
    const float* __restrict__ route_weights,
    const float* __restrict__ mix_in,
    const float* __restrict__ mix_out,
    float* __restrict__ output,
    int64_t tokens,
    int64_t active,
    int64_t experts,
    int64_t rank,
    int64_t hidden_size,
    float hard_route_scale) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= tokens) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_count = (blockDim.x + 31) >> 5;
    extern __shared__ float shared[];
    float* warp_sums = shared;
    float* latent = shared + warp_count;

    // Compute and route-weight each selected group's low-rank latent.  The
    // selected expert index is data, not a host-side branch, so graph replay
    // keeps a fixed launch shape while still touching only K groups.
    for (int64_t slot = 0; slot < active; ++slot) {
        const int64_t expert = selected_ids[token * active + slot];
        if (expert < 0 || expert >= experts) return;
        const int64_t input_base =
            (token * active + slot) * hidden_size;
        for (int64_t component = 0; component < rank; ++component) {
            const int64_t mix_base =
                (expert * rank + component) * hidden_size;
            float partial = 0.0f;
            for (int64_t h = tid; h < hidden_size; h += blockDim.x) {
                partial += selected_outputs[input_base + h]
                    * mix_in[mix_base + h];
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffff, partial, offset);
            }
            if (lane == 0) warp_sums[warp] = partial;
            __syncthreads();
            if (tid == 0) {
                float total = 0.0f;
                for (int w = 0; w < warp_count; ++w) {
                    total += warp_sums[w];
                }
                latent[slot * rank + component] =
                    route_weights[token * active + slot] * total;
            }
            __syncthreads();
        }
    }

    const int64_t output_base = token * hidden_size;
    for (int64_t h = tid; h < hidden_size; h += blockDim.x) {
        float value = 0.0f;
        for (int64_t slot = 0; slot < active; ++slot) {
            const int64_t expert = selected_ids[token * active + slot];
            const int64_t mix_base = (expert * hidden_size + h) * rank;
            for (int64_t component = 0; component < rank; ++component) {
                value += latent[slot * rank + component]
                    * mix_out[mix_base + component];
            }
        }
        output[output_base + h] = hard_route_scale * value;
    }
}

}  // namespace

torch::Tensor qwen_correction_dispatch_cuda(
    torch::Tensor selected_outputs,
    torch::Tensor selected_ids,
    torch::Tensor route_weights,
    torch::Tensor mix_in,
    torch::Tensor mix_out,
    double hard_route_scale) {
    TORCH_CHECK(selected_outputs.scalar_type() == torch::kFloat32,
                "selected outputs must be float32");
    TORCH_CHECK(selected_ids.scalar_type() == torch::kInt64,
                "selected ids must be int64");
    TORCH_CHECK(route_weights.scalar_type() == torch::kFloat32,
                "route weights must be float32");
    TORCH_CHECK(mix_in.scalar_type() == torch::kFloat32,
                "mix_in must be float32");
    TORCH_CHECK(mix_out.scalar_type() == torch::kFloat32,
                "mix_out must be float32");
    TORCH_CHECK(selected_outputs.dim() == 3,
                "selected outputs must be [tokens, active, hidden]");
    TORCH_CHECK(selected_ids.dim() == 2,
                "selected ids must be [tokens, active]");
    TORCH_CHECK(route_weights.sizes() == selected_ids.sizes(),
                "route weights shape mismatch");
    TORCH_CHECK(mix_in.dim() == 3,
                "mix_in must be [experts, rank, hidden]");
    TORCH_CHECK(mix_out.dim() == 3,
                "mix_out must be [experts, hidden, rank]");
    const int64_t tokens = selected_outputs.size(0);
    const int64_t active = selected_outputs.size(1);
    const int64_t hidden_size = selected_outputs.size(2);
    const int64_t experts = mix_in.size(0);
    const int64_t rank = mix_in.size(1);
    TORCH_CHECK(selected_ids.size(0) == tokens,
                "selected ids token dimension mismatch");
    TORCH_CHECK(mix_in.size(2) == hidden_size,
                "mix_in hidden dimension mismatch");
    TORCH_CHECK(mix_out.size(0) == experts,
                "mix_out expert dimension mismatch");
    TORCH_CHECK(mix_out.size(1) == hidden_size,
                "mix_out hidden dimension mismatch");
    TORCH_CHECK(mix_out.size(2) == rank,
                "mix_out rank dimension mismatch");
    auto output = torch::empty({tokens, hidden_size}, selected_outputs.options());
    if (tokens == 0 || active == 0 || hidden_size == 0) {
        return output;
    }
    const int threads = 256;
    const int warp_count = (threads + 31) / 32;
    // Graph capture may run on a non-default current stream. Binding the
    // launch to the current stream is required for the custom op to become a
    // node in that graph instead of executing outside capture.
    const auto stream = at::cuda::getCurrentCUDAStream();
    correction_kernel<<<
        static_cast<unsigned int>(tokens), threads,
        static_cast<size_t>(warp_count + active * rank) * sizeof(float), stream
    >>>(
        selected_outputs.data_ptr<float>(), selected_ids.data_ptr<int64_t>(),
        route_weights.data_ptr<float>(), mix_in.data_ptr<float>(),
        mix_out.data_ptr<float>(), output.data_ptr<float>(), tokens, active,
        experts, rank, hidden_size, static_cast<float>(hard_route_scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
