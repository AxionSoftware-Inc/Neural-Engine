# V0.140 Qwen Bank Isolated FFN Timing

## Question

Is the latency problem caused by the surrounding Qwen model, or is the
selected-token circuit-bank implementation itself slower than the parent FFN?

## Protocol

The V0.138 four-expert top-1 bank is trained with the same settings and
captured Qwen layer-26 hidden states. Instead of timing the full Qwen forward,
the frozen parent MLP and the sparse bank are timed directly on the same four
hidden-state batches, with 20 warmup and 50 synchronized CUDA iterations.
Only the selected expert body is called by the bank. No fused custom CUDA
kernel is used.

## Results

| seed | parent FFN mean ms | sparse bank mean ms | bank/parent | quality gate |
|---:|---:|---:|---:|:---:|
| 2026 | 0.850 | 1.136 | 1.337x | pass |
| 2027 | 0.847 | 1.165 | 1.375x | pass |

The quality gate remains unchanged from V0.138: alpha=0 CE deltas are
`+0.0237` and `+0.0257`. The active expert-body fraction is 25%.

## Interpretation

The slowdown is inside the sparse bank itself. It comes from dispatching
small token subsets through multiple LayerNorm/Linear operations and Python
control flow; it is not caused by Qwen's other layers. Reducing active
parameters therefore does not automatically reduce latency on GPU.

## Decision

**Keep the bank as a quality/architecture GO, but reject the current execution
path as a performance implementation.** The next implementation must pack
expert weights and use grouped batched matmul or a fused CUDA/Triton kernel.
Only after that timing gate passes should more Qwen layers or larger models be
attempted.

## Artifacts

- `benchmark_qwen_single_layer_bank.py`
- `results/runs/qwen_single_layer_bank4x1_ffn_timing_seed2026.json`
- `results/runs/qwen_single_layer_bank4x1_ffn_timing_seed2027.json`
