#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kRowTile = 8;
constexpr int kColTile = 32;
constexpr int kKTile = 32;

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

// One block owns one expert and eight packed token rows.  The gate and value
// projections reuse input tiles, then the output projection reuses the
// resulting SwiGLU coefficients.  The kernel deliberately works on the
// already-packed expert-major buffer: the Python path still owns routing and
// packing, while this probe isolates whether tiled projection is a better
// replacement for three small strided BMMs.
__global__ void tiled_grouped_kernel(
    const float* __restrict__ grouped_hidden,
    const float* __restrict__ gate_weight,   // [E, H, G]
    const float* __restrict__ value_weight,  // [E, H, G]
    const float* __restrict__ output_weight, // [E, G, H]
    float* __restrict__ grouped_output,
    int64_t experts,
    int64_t max_count,
    int64_t hidden_size,
    int64_t group_size) {
    const int64_t row_tiles = (max_count + kRowTile - 1) / kRowTile;
    const int64_t block = static_cast<int64_t>(blockIdx.x);
    const int64_t expert = block / row_tiles;
    const int64_t row_tile = block - expert * row_tiles;
    if (expert >= experts) return;
    const int64_t row_base = row_tile * kRowTile;

    extern __shared__ float shared[];
    float* tile_hidden = shared;
    float* tile_gate = tile_hidden + kRowTile * kKTile;
    float* tile_value = tile_gate + kKTile * kColTile;
    float* coefficient = tile_value + kKTile * kColTile;

    const int tid = threadIdx.x;
    const int row = tid / kColTile;
    const int col = tid % kColTile;
    const bool valid_row = row_base + row < max_count;

    // A tiled gate/value projection.  The contiguous [E,H,G] layout makes
    // each weight tile load coalesced along the output-column dimension.
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
                const int64_t source_row = row_base + local_row;
                const int64_t source_hidden = hidden_base + local_hidden;
                tile_hidden[index] = (
                    source_row < max_count && source_hidden < hidden_size
                ) ? grouped_hidden[
                    (expert * max_count + source_row) * hidden_size
                    + source_hidden
                ] : 0.0f;
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

    // Output projection: each output-column tile reuses a weight tile across
    // the eight rows in this block and accumulates in the same order for all
    // rows.  Padded rows are written too, so the caller needs no extra zeroing.
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
            grouped_output[
                (expert * max_count + row_base + row) * hidden_size
                + hidden_base + col
            ] = output_acc;
        }
    }
}

}  // namespace

torch::Tensor qwen_tiled_grouped_cuda(
    torch::Tensor grouped_hidden,
    torch::Tensor gate_weight,
    torch::Tensor value_weight,
    torch::Tensor output_weight) {
    TORCH_CHECK(grouped_hidden.scalar_type() == torch::kFloat32,
                "grouped hidden must be float32");
    TORCH_CHECK(gate_weight.scalar_type() == torch::kFloat32,
                "gate weights must be float32");
    TORCH_CHECK(value_weight.scalar_type() == torch::kFloat32,
                "value weights must be float32");
    TORCH_CHECK(output_weight.scalar_type() == torch::kFloat32,
                "output weights must be float32");
    TORCH_CHECK(grouped_hidden.dim() == 3,
                "grouped hidden must be [experts, rows, hidden]");
    TORCH_CHECK(gate_weight.dim() == 3 && value_weight.dim() == 3,
                "gate/value weights must be [experts, hidden, group]");
    TORCH_CHECK(output_weight.dim() == 3,
                "output weights must be [experts, group, hidden]");
    TORCH_CHECK(grouped_hidden.is_contiguous() && gate_weight.is_contiguous()
                    && value_weight.is_contiguous()
                    && output_weight.is_contiguous(),
                "tiled grouped inputs must be contiguous");
    const int64_t experts = grouped_hidden.size(0);
    const int64_t max_count = grouped_hidden.size(1);
    const int64_t hidden_size = grouped_hidden.size(2);
    const int64_t group_size = gate_weight.size(2);
    TORCH_CHECK(gate_weight.size(0) == experts
                    && value_weight.size(0) == experts
                    && output_weight.size(0) == experts,
                "expert dimension mismatch");
    TORCH_CHECK(gate_weight.size(1) == hidden_size
                    && value_weight.size(1) == hidden_size
                    && output_weight.size(2) == hidden_size,
                "hidden dimension mismatch");
    TORCH_CHECK(value_weight.sizes() == gate_weight.sizes(),
                "gate/value shape mismatch");
    TORCH_CHECK(output_weight.size(1) == group_size,
                "output group dimension mismatch");
    TORCH_CHECK(hidden_size % kKTile == 0 && group_size % kKTile == 0,
                "tiled grouped kernel requires dimensions divisible by 32");
    TORCH_CHECK(max_count > 0 && experts > 0,
                "tiled grouped kernel requires non-empty inputs");

    auto grouped_output = torch::empty_like(grouped_hidden);
    const int64_t row_tiles = (max_count + kRowTile - 1) / kRowTile;
    const auto blocks = static_cast<unsigned int>(experts * row_tiles);
    const size_t shared_bytes = static_cast<size_t>(
        kRowTile * kKTile + 2 * kKTile * kColTile
        + kRowTile * group_size) * sizeof(float);
    const auto stream = at::cuda::getCurrentCUDAStream();
    tiled_grouped_kernel<<<blocks, kThreads, shared_bytes, stream>>>(
        grouped_hidden.data_ptr<float>(), gate_weight.data_ptr<float>(),
        value_weight.data_ptr<float>(), output_weight.data_ptr<float>(),
        grouped_output.data_ptr<float>(), experts, max_count, hidden_size,
        group_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return grouped_output;
}
