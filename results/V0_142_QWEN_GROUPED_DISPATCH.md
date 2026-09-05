# V0.142 Qwen Grouped Dispatch

## Question

Can grouping selected tokens by expert, then using one padded batched matrix
multiplication per projection, remove enough dispatch overhead to make the
sparse bank faster than the parent Qwen FFN?

## Protocol

The V0.138 four-expert top-1 single-layer Qwen3-0.6B bank is repeated with the
same layer, data, training steps, and quality gate. For each hard-routed
selected token, the implementation first sorts pairs by expert, reuses one
expert weight matrix for its token block, and runs grouped batched matmuls for
the input and output projections. Padding handles uneven expert loads. Timing
uses 20 CUDA warmup iterations and 50 synchronized serial iterations on both
the captured layer-26 FFN and the full Qwen forward.

## Results

| seed | parent FFN mean ms | grouped bank mean ms | bank/parent | end-to-end bank/parent | quality gate |
|---:|---:|---:|---:|---:|:---:|
| 2026 | 0.848 | 0.989 | 1.166x | 1.019x | pass |
| 2027 | 0.851 | 1.025 | 1.204x | 1.016x | pass |

The quality signal remains the V0.138 result: alpha=0 CE deltas are `+0.0237`
and `+0.0257`, with teacher top-1 agreement `96.39%` and `95.51%`. The active
expert-body fraction is 25%.

Grouped dispatch is a clear implementation improvement over V0.140's
token-loop (`1.337x` and `1.375x` isolated FFN ratios), and it avoids V0.141's
per-token `bmm` penalty (`about 9.90x`). It still does not beat the dense
parent FFN.

## Interpretation

The architecture continues to pass the local quality gate, but the current
Python/PyTorch grouped implementation is still not a performance win. Sorting,
padding, index movement, expert-specific LayerNorm, and two grouped launches
consume more time than the parent projection at this small batch and sequence
size.

## Decision

**Accept grouped dispatch as the best current reference implementation, but do
not claim a speedup.** The next focused test is to remove the child's internal
LayerNorm when the surrounding Qwen block already supplies normalized input,
then remeasure both quality and latency. If that does not close the gap, a
real fused grouped CUDA/Triton kernel is required before scaling model size or
layer count.

## Artifacts

- `benchmark_qwen_two_layer_transplant.py`
- `benchmark_qwen_single_layer_bank.py`
- `results/runs/qwen_single_layer_bank4x1_grouped_timing_seed2026.json`
- `results/runs/qwen_single_layer_bank4x1_grouped_timing_seed2027.json`
