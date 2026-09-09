# V0.252 — In-place fused SwiGLU intermediate audit

**Date:** 2026-09-09  
**Status:** `REJECTED; NO MATERIAL RUNTIME GAIN`  
**Branch:** `exp/track-runtime`

## Question

The fused grouped projection first computes a concatenated gate/value BMM,
then applies SiLU and multiplies the two halves. This probe reuses the gate
half in-place for those two operations, aiming to remove intermediate tensor
allocation and kernels. Routing, expert weights, projection math, correction,
and output contract are unchanged.

The mode is exposed as the opt-in benchmark path
`grouped-adaptive-fixed-pack-inplace`; the normal fixed-pack path remains the
control.

## Protocol

Local `Qwen/Qwen3-0.6B`, layers `19--26`, `E=8`, `K=5`, rank-1 correction,
float32 CUDA, and trained-cascade recipe `600/600/200`. Prefix length is 4;
batch sizes are 1, 8, and 32. The screen uses 5 CUDA-Graph warmups and 10
timing iterations at each shape, with seeds 2026 and 17.

## CUDA-Graph latency

Values are milliseconds; lower is better.

| seed | path | B1 | B8 | B32 |
|---:|---|---:|---:|---:|
| 2026 | fixed-pack | 10.738 | 16.525 | 33.849 |
| 2026 | in-place | 10.758 | 16.525 | 33.933 |
| 17 | fixed-pack | 10.879 | 16.435 | 33.804 |
| 17 | in-place | 10.755 | 16.599 | 33.825 |
| **two-seed mean** | **fixed-pack** | **10.809** | **16.480** | **33.827** |
| **two-seed mean** | **in-place** | **10.757** | **16.562** | **33.879** |
| **in-place change** | **vs fixed-pack** | **-0.48%** | **+0.50%** | **+0.16%** |

The B1 improvement is not repeated at B8/B32 and is within timing noise. The
variant does not provide a stable speed tier, so no long confirmation run is
warranted.

## Correctness and quality

- Both seeds passed the strict graph/eager numerical gate (`max error` about
  `1.2e-5` to `1.4e-5`, tolerance `1e-3`).
- Eight-token greedy generation matched the grouped baseline exactly for both
  seeds.
- The quality recipe is unchanged; CE deltas versus dense were `+0.047010`
  and `+0.038573`. No quality improvement is claimed.

Raw records:

- `runs/v0_252_inplace_swiglu_seed2026.json`
- `runs/v0_252_inplace_swiglu_seed17.json`

## Decision

Reject in-place SwiGLU as a serving optimization. Keep the mode only as a
reproducible opt-in ablation; do not use it in the default path. The remaining
runtime work should target grouped-GEMM scheduling/reuse rather than local
activation-buffer mechanics.
