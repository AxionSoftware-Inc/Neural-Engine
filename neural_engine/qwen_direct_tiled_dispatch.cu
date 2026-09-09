#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kRowTile = 8;
constexpr int kColTile = 32;
constexpr int kKTile = 32;

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

// One block owns an expert and eight expert-local rows.  Unlike the existing
// grouped tiled probe, the input rows are gathered directly from flat_hidden
// using sorted route metadata.  The output is atomically accumulated into
// token-major storage, so packing and grouped-output finalization disappear.
__global__ void direct_tiled_kernel(
    const float* __restrict__ flat_hidden,
    const int64_t* __restrict__ sorted_token_ids,
    const int64_t* __restrict__ sorted_slots,
    const int64_t* __restrict__ starts,
    const int64_t* __restrict__ counts,
    const float* __restrict__ route_weights,
    const float* __restrict__ gate_weight,   // [E, H, G]
    const float* __restrict__ value_weight,  // [E, H, G]
    const float* __restrict__ output_weight, // [E, G, H]
    float* __restrict__ output,
    int64_t experts,
    int64_t tokens,
    int64_t pairs,
    int64_t hidden_size,
    int64_t group_size,
    int64_t active,
    float hard_route_scale,
    bool uniform_accum) {
    const int64_t row_tiles = (tokens + kRowTile - 1) / kRowTile;
    const int64_t block = static_cast<int64_t>(blockIdx.x);
    const int64_t expert = block / row_tiles;
    const int64_t row_tile = block - expert * row_tiles;
    if (expert >= experts) return;
    const int64_t row_base = row_tile * kRowTile;

    __shared__ int64_t tile_tokens[kRowTile];
    __shared__ int64_t tile_slots[kRowTile];
    extern __shared__ float shared[];
    float* tile_hidden = shared;
    float* tile_gate = tile_hidden + kRowTile * kKTile;
    float* tile_value = tile_gate + kKTile * kColTile;
    float* coefficient = tile_value + kKTile * kColTile;

    const int tid = threadIdx.x;
    const int row = tid / kColTile;
    const int col = tid % kColTile;
    const int64_t expert_start = starts[expert];
    const int64_t expert_count = counts[expert];
    const bool valid_row = row_base + row < expert_count
        && expert_start + row_base + row < pairs;
    if (tid < kRowTile) {
        const int64_t sorted_index = expert_start + row_base + tid;
        const bool tile_row_valid = row_base + tid < expert_count
            && sorted_index < pairs;
        tile_tokens[tid] = tile_row_valid ? sorted_token_ids[sorted_index] : -1;
        tile_slots[tid] = tile_row_valid ? sorted_slots[sorted_index] : 0;
    }
    __syncthreads();

    for (int64_t group_base = 0; group_base < group_size;
         group_base += kColTile) {
        float gate_acc = 0.0f;
        float value_acc = 0.0f;
        for (int64_t hidden_base = 0; hidden_base < hidden_size;
             hidden_base += kKTile) {
            for (int index = tid; index < kRowTile * kKTile;
                 index += kThreads) {
                const int local_row = index / kKTile;
                const int local_hidden = index % kKTile;
                const int64_t source_hidden = hidden_base + local_hidden;
                const int64_t token = tile_tokens[local_row];
                tile_hidden[index] = (
                    token >= 0 && source_hidden < hidden_size
                ) ? flat_hidden[token * hidden_size + source_hidden] : 0.0f;
            }
            for (int index = tid; index < kKTile * kColTile;
                 index += kThreads) {
                const int local_hidden = index / kColTile;
                const int local_col = index % kColTile;
                const int64_t source_hidden = hidden_base + local_hidden;
                const int64_t source_group = group_base + local_col;
                if (source_hidden < hidden_size && source_group < group_size) {
                    tile_gate[index] = gate_weight[
                        (expert * hidden_size + source_hidden) * group_size
                        + source_group
                    ];
                    tile_value[index] = value_weight[
                        (expert * hidden_size + source_hidden) * group_size
                        + source_group
                    ];
                } else {
                    tile_gate[index] = 0.0f;
                    tile_value[index] = 0.0f;
                }
            }
            __syncthreads();
            if (row < kRowTile && col < kColTile && valid_row
                && group_base + col < group_size) {
                for (int local_hidden = 0; local_hidden < kKTile;
                     ++local_hidden) {
                    gate_acc += tile_hidden[row * kKTile + local_hidden]
                        * tile_gate[local_hidden * kColTile + col];
                    value_acc += tile_hidden[row * kKTile + local_hidden]
                        * tile_value[local_hidden * kColTile + col];
                }
            }
            __syncthreads();
        }
        if (row < kRowTile && col < kColTile && valid_row
            && group_base + col < group_size) {
            coefficient[row * group_size + group_base + col] =
                silu(gate_acc) * value_acc;
        }
        __syncthreads();
    }

    for (int64_t hidden_base = 0; hidden_base < hidden_size;
         hidden_base += kColTile) {
        float output_acc = 0.0f;
        for (int64_t group_base = 0; group_base < group_size;
             group_base += kKTile) {
            for (int index = tid; index < kKTile * kColTile;
                 index += kThreads) {
                const int local_group = index / kColTile;
                const int local_hidden = index % kColTile;
                const int64_t source_group = group_base + local_group;
                const int64_t source_hidden = hidden_base + local_hidden;
                tile_gate[index] = (
                    source_group < group_size && source_hidden < hidden_size
                ) ? output_weight[
                    (expert * group_size + source_group) * hidden_size
                    + source_hidden
                ] : 0.0f;
            }
            __syncthreads();
            if (row < kRowTile && col < kColTile && valid_row
                && hidden_base + col < hidden_size) {
                for (int local_group = 0; local_group < kKTile;
                     ++local_group) {
                    output_acc += coefficient[row * group_size
                                              + group_base + local_group]
                        * tile_gate[local_group * kColTile + col];
                }
            }
            __syncthreads();
        }
        if (row < kRowTile && col < kColTile && valid_row
            && hidden_base + col < hidden_size) {
            const int64_t token = tile_tokens[row];
            const int64_t slot = tile_slots[row];
            const float route = uniform_accum
                ? (1.0f / static_cast<float>(active))
                : route_weights[token * active + slot];
            atomicAdd(
                output + token * hidden_size + hidden_base + col,
                hard_route_scale * route * output_acc);
        }
    }
}

}  // namespace

torch::Tensor qwen_direct_tiled_cuda(
    torch::Tensor flat_hidden,
    torch::Tensor sorted_token_ids,
    torch::Tensor sorted_slots,
    torch::Tensor starts,
    torch::Tensor counts,
    torch::Tensor route_weights,
    torch::Tensor group_gate_weight,
    torch::Tensor group_value_weight,
    torch::Tensor group_output_weight,
    int64_t active_experts,
    double hard_route_scale,
    bool uniform_accum) {
    TORCH_CHECK(flat_hidden.scalar_type() == torch::kFloat32,
                "hidden must be float32");
    TORCH_CHECK(route_weights.scalar_type() == torch::kFloat32,
                "route weights must be float32");
    TORCH_CHECK(group_gate_weight.scalar_type() == torch::kFloat32
                    && group_value_weight.scalar_type() == torch::kFloat32
                    && group_output_weight.scalar_type() == torch::kFloat32,
                "weights must be float32");
    TORCH_CHECK(sorted_token_ids.scalar_type() == torch::kInt64
                    && sorted_slots.scalar_type() == torch::kInt64
                    && starts.scalar_type() == torch::kInt64
                    && counts.scalar_type() == torch::kInt64,
                "route metadata must be int64");
    TORCH_CHECK(flat_hidden.dim() == 2 && route_weights.dim() == 2,
                "hidden and routes must be rank-2");
    TORCH_CHECK(group_gate_weight.dim() == 3
                    && group_value_weight.dim() == 3
                    && group_output_weight.dim() == 3,
                "weights must be rank-3");
    TORCH_CHECK(flat_hidden.is_contiguous() && sorted_token_ids.is_contiguous()
                    && sorted_slots.is_contiguous() && starts.is_contiguous()
                    && counts.is_contiguous() && route_weights.is_contiguous()
                    && group_gate_weight.is_contiguous()
                    && group_value_weight.is_contiguous()
                    && group_output_weight.is_contiguous(),
                "direct tiled inputs must be contiguous");
    const int64_t tokens = flat_hidden.size(0);
    const int64_t hidden_size = flat_hidden.size(1);
    const int64_t pairs = sorted_token_ids.size(0);
    const int64_t experts = group_gate_weight.size(0);
    const int64_t group_size = group_gate_weight.size(2);
    TORCH_CHECK(active_experts > 0 && active_experts <= route_weights.size(1),
                "invalid active expert count");
    TORCH_CHECK(sorted_slots.size(0) == pairs && starts.size(0) == experts
                    && counts.size(0) == experts,
                "route metadata shape mismatch");
    TORCH_CHECK(group_value_weight.sizes() == group_gate_weight.sizes(),
                "gate/value shape mismatch");
    TORCH_CHECK(group_gate_weight.size(1) == hidden_size
                    && group_output_weight.size(0) == experts
                    && group_output_weight.size(1) == group_size
                    && group_output_weight.size(2) == hidden_size,
                "weight shape mismatch");
    TORCH_CHECK(hidden_size % kKTile == 0 && group_size % kKTile == 0,
                "direct tiled kernel requires dimensions divisible by 32");
    auto output = torch::zeros({tokens, hidden_size}, flat_hidden.options());
    if (tokens == 0 || pairs == 0 || experts == 0) return output;
    const int64_t row_tiles = (tokens + kRowTile - 1) / kRowTile;
    const auto blocks = static_cast<unsigned int>(experts * row_tiles);
    const size_t shared_bytes = static_cast<size_t>(
        kRowTile * kKTile + 2 * kKTile * kColTile
        + kRowTile * group_size) * sizeof(float);
    const auto stream = at::cuda::getCurrentCUDAStream();
    direct_tiled_kernel<<<blocks, kThreads, shared_bytes, stream>>>(
        flat_hidden.data_ptr<float>(), sorted_token_ids.data_ptr<int64_t>(),
        sorted_slots.data_ptr<int64_t>(), starts.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(), route_weights.data_ptr<float>(),
        group_gate_weight.data_ptr<float>(),
        group_value_weight.data_ptr<float>(),
        group_output_weight.data_ptr<float>(), output.data_ptr<float>(),
        experts, tokens, pairs, hidden_size, group_size, active_experts,
        static_cast<float>(hard_route_scale), uniform_accum);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
