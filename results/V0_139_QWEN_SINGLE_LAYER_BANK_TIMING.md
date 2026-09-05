# V0.139 Qwen Single-Layer Bank Timing

## Question

Does the 4-expert top-1 single-layer circuit bank produce an actual end-to-end
GPU speedup when only the selected expert body is evaluated?

## Protocol

The V0.138 single-layer Qwen3-0.6B bank is repeated with identical model,
layer, data, and training settings. After training, the frozen parent FFN and
the hard top-1 bank are each measured for 20 warmup and 50 synchronized CUDA
iterations over the same four evaluation batches. The bank dispatches only
selected token-expert pairs; no custom fused CUDA kernel is used.

## Results

| seed | parent mean ms | sparse-bank mean ms | bank/parent | quality gate |
|---:|---:|---:|---:|:---:|
| 2026 | 66.70 | 68.00 | 1.020x | pass |
| 2027 | 66.86 | 68.25 | 1.021x | pass |

The corresponding alpha=0 quality results remain CE deltas `+0.0237` and
`+0.0257`, with teacher top-1 agreement `96.39%` and `95.51%`. The bank still
executes one of four expert bodies per token (25% expert-body fraction), but
the current Python-level grouping adds enough overhead to erase the theoretical
compute reduction.

## Interpretation

The sparse circuit bank is a quality GO but the current execution
implementation is not a latency GO. The result is consistent across isolated
serial runs; the earlier parallel timing outlier is discarded as GPU
contention. This separates the architecture question from the kernel/layout
question: selected-token routing works, while its implementation needs a
batched or fused dispatch path.

## Decision

**Keep the bank and reject the current Python dispatch for performance.** The
next gate is grouped/batched CUDA-friendly execution with the same outputs and
quality threshold. Do not increase model size or add more layers until the
implementation shows a repeatable speed or memory advantage.

## Artifacts

- `benchmark_qwen_single_layer_bank.py`
- `results/runs/qwen_single_layer_bank4x1_timing_serial_seed2026.json`
- `results/runs/qwen_single_layer_bank4x1_timing_serial_seed2027.json`
