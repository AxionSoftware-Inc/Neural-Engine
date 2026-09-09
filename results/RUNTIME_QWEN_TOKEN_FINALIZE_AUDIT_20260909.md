# V0.253 — Token-owned atomics-free grouped finalizer audit

**Date:** 2026-09-09  
**Status:** `CORRECT; NO MATERIAL RUNTIME GAIN; OPT-IN REFERENCE`  
**Branch:** `exp/track-runtime`

## Question

The grouped folded-output finalizer previously launched one block per selected
route and used `atomicAdd` because K routes contribute to each token. This
probe assigns one block to each token, reduces its K selected outputs in fixed
slot order, and writes the token once. The goal is to remove atomic contention
without changing routing, projections, correction, or accumulation math.

The mode is exposed as the opt-in path
`grouped-adaptive-fixed-pack-token-finalize`; fixed-pack's existing atomic
finalizer is the control.

## Protocol

Local `Qwen/Qwen3-0.6B`, layers `19--26`, `E=8`, `K=5`, rank-1 correction,
float32 CUDA, and trained-cascade recipe `600/600/200`. Prefix length is 4;
batch sizes are 1, 8, and 32. The screen uses 5 CUDA-Graph warmups and 10
timing iterations at each shape, with seeds 2026 and 17.

## CUDA-Graph latency

Values are milliseconds; lower is better.

| seed | fixed-pack | token-owned finalizer | token change |
|---:|---:|---:|---:|
| 2026 B1 | 10.686 | 10.736 | +0.47% |
| 2026 B8 | 16.523 | 16.482 | -0.25% |
| 2026 B32 | 33.861 | 33.617 | -0.72% |
| 17 B1 | 10.739 | 10.721 | -0.17% |
| 17 B8 | 16.596 | 16.589 | -0.04% |
| 17 B32 | 33.738 | 33.984 | +0.73% |
| **two-seed mean B1** | **10.713** | **10.729** | **+0.15%** |
| **two-seed mean B8** | **16.560** | **16.536** | **-0.15%** |
| **two-seed mean B32** | **33.800** | **33.801** | **+0.00%** |

The sign flips at B32 across seeds and all two-seed changes are below a
material threshold. Removing atomics does not create a new runtime tier on
this GPU and shape range.

## Correctness and quality

- Both seeds passed the strict graph/eager numerical gate; maximum errors were
  about `1.3e-5` against the `1e-3` tolerance.
- A random CUDA smoke test matched a PyTorch reference reduction with maximum
  absolute error `4.8e-7`.
- Eight-token greedy generation matched the grouped baseline exactly for both
  seeds.
- Quality is unchanged by this inference-only backend change; CE deltas
  versus dense were `+0.045470` and `+0.038005`.

Raw records:

- `runs/v0_253_token_owned_finalize_seed2026.json`
- `runs/v0_253_token_owned_finalize_seed17.json`

## Decision

Keep the token-owned finalizer as a correctness/reference implementation, but
do not make it the default or spend a long run on it. It is practically tied
with the existing atomic finalizer. The next runtime work should target the
pack/projection boundary and grouped-GEMM scheduling, where a material gain is
still possible.
