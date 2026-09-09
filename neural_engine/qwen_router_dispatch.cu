#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <tuple>

namespace {

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

// One block owns one token.  It evaluates the two small router projections,
// performs a deterministic descending top-k, and normalizes only the chosen
// logits.  This removes the separate Linear/SiLU/Linear/topk/softmax launches
// for fixed-shape decode.  The kernel is intentionally opt-in: its reduction
// order and tie ordering are audited against the PyTorch route before use.
__global__ void router_kernel(
    const float* __restrict__ hidden,
    const float* __restrict__ first_weight,
    const float* __restrict__ first_bias,
    const float* __restrict__ second_weight,
    const float* __restrict__ second_bias,
    int64_t* __restrict__ selected_ids,
    float* __restrict__ route_weights,
    int64_t tokens,
    int64_t hidden_size,
    int64_t router_size,
    int64_t experts,
    int64_t active,
    float temperature) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= tokens) return;
    extern __shared__ float shared[];
    float* intermediate = shared;
    float* scores = intermediate + router_size;
    float* top_values = scores + experts;
    const int64_t aligned_active = (active + 1) & ~static_cast<int64_t>(1);
    int64_t* top_ids = reinterpret_cast<int64_t*>(top_values + aligned_active);

    const float* hidden_row = hidden + token * hidden_size;
    for (int64_t r = threadIdx.x; r < router_size; r += blockDim.x) {
        float value = first_bias[r];
        const float* weight_row = first_weight + r * hidden_size;
        for (int64_t h = 0; h < hidden_size; ++h) {
            value += hidden_row[h] * weight_row[h];
        }
        intermediate[r] = silu(value);
    }
    __syncthreads();

    for (int64_t expert = threadIdx.x; expert < experts; expert += blockDim.x) {
        float value = second_bias[expert];
        const float* weight_row = second_weight + expert * router_size;
        for (int64_t r = 0; r < router_size; ++r) {
            value += intermediate[r] * weight_row[r];
        }
        scores[expert] = value;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int64_t slot = 0; slot < active; ++slot) {
            float best_value = -FLT_MAX;
            int64_t best_id = -1;
            for (int64_t expert = 0; expert < experts; ++expert) {
                bool already_selected = false;
                for (int64_t previous = 0; previous < slot; ++previous) {
                    if (top_ids[previous] == expert) {
                        already_selected = true;
                        break;
                    }
                }
                if (!already_selected && scores[expert] > best_value) {
                    best_value = scores[expert];
                    best_id = expert;
                }
            }
            top_values[slot] = best_value;
            top_ids[slot] = best_id;
        }
        float maximum = top_values[0] / temperature;
        for (int64_t slot = 1; slot < active; ++slot) {
            maximum = fmaxf(maximum, top_values[slot] / temperature);
        }
        float denominator = 0.0f;
        for (int64_t slot = 0; slot < active; ++slot) {
            denominator += expf(top_values[slot] / temperature - maximum);
        }
        for (int64_t slot = 0; slot < active; ++slot) {
            selected_ids[token * active + slot] = top_ids[slot];
            route_weights[token * active + slot] = expf(
                top_values[slot] / temperature - maximum
            ) / denominator;
        }
    }
}

__global__ void subset_router_kernel(
    const float* __restrict__ hidden,
    const float* __restrict__ first_weight,
    const float* __restrict__ first_bias,
    const float* __restrict__ second_weight,
    const float* __restrict__ second_bias,
    const float* __restrict__ subset_membership,
    int64_t* __restrict__ selected_ids,
    float* __restrict__ route_weights,
    int64_t tokens,
    int64_t hidden_size,
    int64_t router_size,
    int64_t subsets,
    int64_t experts,
    int64_t active) {
    const int64_t token = static_cast<int64_t>(blockIdx.x);
    if (token >= tokens) return;
    extern __shared__ float shared[];
    float* intermediate = shared;
    float* scores = intermediate + router_size;
    const float* hidden_row = hidden + token * hidden_size;
    for (int64_t r = threadIdx.x; r < router_size; r += blockDim.x) {
        float value = first_bias[r];
        const float* weight_row = first_weight + r * hidden_size;
        for (int64_t h = 0; h < hidden_size; ++h) {
            value += hidden_row[h] * weight_row[h];
        }
        intermediate[r] = silu(value);
    }
    __syncthreads();
    for (int64_t subset = threadIdx.x; subset < subsets; subset += blockDim.x) {
        float value = second_bias[subset];
        const float* weight_row = second_weight + subset * router_size;
        for (int64_t r = 0; r < router_size; ++r) {
            value += intermediate[r] * weight_row[r];
        }
        scores[subset] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float best_value = -FLT_MAX;
        int64_t best_subset = 0;
        for (int64_t subset = 0; subset < subsets; ++subset) {
            if (scores[subset] > best_value) {
                best_value = scores[subset];
                best_subset = subset;
            }
        }
        int64_t slot = 0;
        for (int64_t expert = 0; expert < experts && slot < active; ++expert) {
            if (subset_membership[best_subset * experts + expert] > 0.5f) {
                selected_ids[token * active + slot] = expert;
                route_weights[token * active + slot] = 1.0f / static_cast<float>(active);
                ++slot;
            }
        }
        for (; slot < active; ++slot) {
            selected_ids[token * active + slot] = 0;
            route_weights[token * active + slot] = 1.0f / static_cast<float>(active);
        }
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> qwen_router_dispatch_cuda(
    torch::Tensor hidden,
    torch::Tensor first_weight,
    torch::Tensor first_bias,
    torch::Tensor second_weight,
    torch::Tensor second_bias,
    int64_t active_experts,
    double temperature) {
    TORCH_CHECK(hidden.scalar_type() == torch::kFloat32, "hidden must be float32");
    TORCH_CHECK(first_weight.scalar_type() == torch::kFloat32, "first weight must be float32");
    TORCH_CHECK(first_bias.scalar_type() == torch::kFloat32, "first bias must be float32");
    TORCH_CHECK(second_weight.scalar_type() == torch::kFloat32, "second weight must be float32");
    TORCH_CHECK(second_bias.scalar_type() == torch::kFloat32, "second bias must be float32");
    TORCH_CHECK(hidden.dim() == 2, "hidden must be [tokens, hidden]");
    TORCH_CHECK(first_weight.dim() == 2, "first weight must be [router, hidden]");
    TORCH_CHECK(second_weight.dim() == 2, "second weight must be [experts, router]");
    TORCH_CHECK(first_bias.dim() == 1, "first bias must be [router]");
    TORCH_CHECK(second_bias.dim() == 1, "second bias must be [experts]");
    const int64_t tokens = hidden.size(0);
    const int64_t hidden_size = hidden.size(1);
    const int64_t router_size = first_weight.size(0);
    const int64_t experts = second_weight.size(0);
    TORCH_CHECK(first_weight.size(1) == hidden_size, "first hidden mismatch");
    TORCH_CHECK(first_bias.size(0) == router_size, "first bias mismatch");
    TORCH_CHECK(second_weight.size(1) == router_size, "second router mismatch");
    TORCH_CHECK(second_bias.size(0) == experts, "second bias mismatch");
    TORCH_CHECK(active_experts >= 1 && active_experts <= experts, "invalid active count");
    TORCH_CHECK(tokens <= 65535, "token count exceeds fixed one-token launch bound");
    TORCH_CHECK(temperature > 0.0, "temperature must be positive");
    auto selected_ids = torch::empty(
        {tokens, active_experts},
        hidden.options().dtype(torch::kInt64)
    );
    auto route_weights = torch::empty({tokens, active_experts}, hidden.options());
    if (tokens == 0) {
        return std::make_tuple(selected_ids, route_weights);
    }
    const int threads = 256;
    const int64_t aligned_active = (active_experts + 1) & ~static_cast<int64_t>(1);
    const size_t shared_bytes = static_cast<size_t>(router_size + experts + aligned_active)
        * sizeof(float) + static_cast<size_t>(active_experts) * sizeof(int64_t);
    const auto stream = at::cuda::getCurrentCUDAStream();
    router_kernel<<<static_cast<unsigned int>(tokens), threads, shared_bytes, stream>>>(
        hidden.data_ptr<float>(), first_weight.data_ptr<float>(),
        first_bias.data_ptr<float>(), second_weight.data_ptr<float>(),
        second_bias.data_ptr<float>(), selected_ids.data_ptr<int64_t>(),
        route_weights.data_ptr<float>(), tokens, hidden_size, router_size,
        experts, active_experts, static_cast<float>(temperature)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(selected_ids, route_weights);
}

std::tuple<torch::Tensor, torch::Tensor> qwen_subset_router_dispatch_cuda(
    torch::Tensor hidden,
    torch::Tensor first_weight,
    torch::Tensor first_bias,
    torch::Tensor second_weight,
    torch::Tensor second_bias,
    torch::Tensor subset_membership,
    int64_t active_experts) {
    TORCH_CHECK(hidden.scalar_type() == torch::kFloat32, "hidden must be float32");
    TORCH_CHECK(first_weight.scalar_type() == torch::kFloat32, "first weight must be float32");
    TORCH_CHECK(first_bias.scalar_type() == torch::kFloat32, "first bias must be float32");
    TORCH_CHECK(second_weight.scalar_type() == torch::kFloat32, "second weight must be float32");
    TORCH_CHECK(second_bias.scalar_type() == torch::kFloat32, "second bias must be float32");
    TORCH_CHECK(subset_membership.scalar_type() == torch::kFloat32, "subset membership must be float32");
    TORCH_CHECK(hidden.dim() == 2, "hidden must be [tokens, hidden]");
    TORCH_CHECK(first_weight.dim() == 2 && first_bias.dim() == 1, "invalid first projection");
    TORCH_CHECK(second_weight.dim() == 2 && second_bias.dim() == 1, "invalid second projection");
    TORCH_CHECK(subset_membership.dim() == 2, "subset membership must be [subsets, experts]");
    const int64_t tokens = hidden.size(0);
    const int64_t hidden_size = hidden.size(1);
    const int64_t router_size = first_weight.size(0);
    const int64_t subsets = second_weight.size(0);
    const int64_t experts = subset_membership.size(1);
    TORCH_CHECK(first_weight.size(1) == hidden_size, "first hidden mismatch");
    TORCH_CHECK(first_bias.size(0) == router_size, "first bias mismatch");
    TORCH_CHECK(second_weight.size(1) == router_size, "second router mismatch");
    TORCH_CHECK(second_bias.size(0) == subsets, "second bias mismatch");
    TORCH_CHECK(subset_membership.size(0) == subsets, "subset count mismatch");
    TORCH_CHECK(active_experts >= 1 && active_experts <= experts, "invalid active count");
    TORCH_CHECK(tokens <= 65535, "token count exceeds fixed one-token launch bound");
    auto selected_ids = torch::empty(
        {tokens, active_experts}, hidden.options().dtype(torch::kInt64)
    );
    auto route_weights = torch::empty({tokens, active_experts}, hidden.options());
    if (tokens == 0) {
        return std::make_tuple(selected_ids, route_weights);
    }
    const int threads = 256;
    const size_t shared_bytes = static_cast<size_t>(router_size + subsets) * sizeof(float);
    const auto stream = at::cuda::getCurrentCUDAStream();
    subset_router_kernel<<<static_cast<unsigned int>(tokens), threads, shared_bytes, stream>>>(
        hidden.data_ptr<float>(), first_weight.data_ptr<float>(),
        first_bias.data_ptr<float>(), second_weight.data_ptr<float>(),
        second_bias.data_ptr<float>(), subset_membership.data_ptr<float>(),
        selected_ids.data_ptr<int64_t>(), route_weights.data_ptr<float>(),
        tokens, hidden_size, router_size, subsets, experts, active_experts
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(selected_ids, route_weights);
}
