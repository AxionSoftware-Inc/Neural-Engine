# V0.141 Qwen Packed Dispatch

## Question

Can a simple packed/vectorized implementation remove the small-operation
overhead measured in V0.140, while preserving the exact selected-token sparse
route?

## Protocol

The V0.138 four-expert top-1 single-layer Qwen3-0.6B bank is repeated with the
same layer, data, training steps, and quality gate. The new `packed` mode stacks
expert weights, gathers the selected expert for every token, and evaluates the
per-token projections with batched matrix multiplication. It is compared with
the existing token-loop implementation for numerical correctness, then timed
on the captured layer-26 hidden states and end-to-end Qwen forward. Timing uses
20 CUDA warmup iterations and synchronized serial measurements. No custom
CUDA/Triton kernel is used.

The packed and token-loop implementations agree to within `8.94e-8` maximum
absolute error on the same random test input, and both execute only the
selected expert body (25% expected expert-body fraction).

## Results

| seed | parent FFN mean ms | packed bank mean ms | bank/parent | end-to-end bank/parent | quality gate |
|---:|---:|---:|---:|---:|:---:|
| 2026 | 0.846 | 8.377 | 9.907x | 1.124x | pass |
| 2027 | 0.843 | 8.339 | 9.896x | 1.123x | pass |

The quality signal is unchanged from V0.138: alpha=0 CE deltas are `+0.0237`
and `+0.0257`, with teacher top-1 agreement `96.39%` and `95.51%`.

## Interpretation

Naive packing is substantially worse than the token-loop baseline. Although
the route is mathematically correct, per-token gathered matrices create many
small batched matrix operations and extra gather/index-add traffic. The
theoretical reduction in active parameters still does not translate into GPU
latency reduction.

## Decision

**Reject this packed implementation as a performance path and keep it out of
the default.** The default remains the numerically validated token-loop mode;
the packed mode is retained only as an explicit research flag. The architecture
quality result remains a GO, but the execution result is a NO-GO. The next
performance experiment must reuse weights across grouped tokens, using a true
grouped/fused CUDA or Triton kernel rather than materializing a separate matrix
for every token. Do not scale to more Qwen layers or a larger model until that
kernel gate is measured.

## Artifacts

- `benchmark_qwen_two_layer_transplant.py`
- `benchmark_qwen_single_layer_bank.py`
- `results/runs/qwen_single_layer_bank4x1_packed_timing_seed2026.json`
- `results/runs/qwen_single_layer_bank4x1_packed_timing_seed2027.json`
