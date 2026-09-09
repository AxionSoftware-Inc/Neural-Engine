# V0.211 — Custom CUDA correction-kernel audit

**Date:** 2026-09-09  
**Status:** `PARITY PASS; NO LARGE SPEEDUP; OPT-IN ONLY`  
**Branch:** `exp/track-runtime`

## Question

The trained K=5 child still pays for selected low-rank correction dispatch.
This experiment adds a fixed-shape CUDA kernel that computes only the selected
K groups for each token, with no host-side expert loop. It is compared with the
existing vectorized PyTorch correction on the same trained rank-4 child.

## Implementation issue found and fixed

The first kernel launch used the ATen default stream. CUDA Graph capture in this
path uses the current capture stream, so the operation was not recorded and the
graph appeared empty. The kernel now uses `getCurrentCUDAStream()`. A standalone
graph smoke then passes with zero replay difference.

## Trained B8 results

| backend | eager | graph | graph / eager | final graph-vs-eager parity | vs vectorized |
|---|---:|---:|---:|---:|---|
| vectorized | `44.477 ms` | `49.359 ms` | `1.110x` | `8.58e-6` | reference |
| custom CUDA | `44.778 ms` | `48.881 ms` | `1.092x` | `9.78e-6` | eager `+0.7%`, graph `−1.0%` |

Custom CUDA final logits also match the vectorized backend within
`9.30e-6` eager and `9.54e-6` graph. The overall audit remains `PARITY_PASS`;
greedy generation and the existing trained quality result are unchanged.

The small timing movement is within ordinary GPU measurement noise and is not
a meaningful end-to-end breakthrough. The kernel does establish a graph-safe
selected-correction primitive, but the full B8 sparse path still includes base
projection, routing, cache, and launch overhead.

## Decision

- retain the custom CUDA correction kernel as an opt-in backend;
- keep vectorized correction as the default;
- do not claim a quality or large latency improvement from this patch;
- target a fused base-projection plus correction dispatch next, or another
  kernel that removes both selected-group and correction launch boundaries.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --seed 2026 --warmup 40 --iterations 100 --correction-backend-iterations 30 --single-token-backend-iterations 30 --compiled-child-iterations 2 --batch-sizes 1 8 --output results/runs/v0_211_rank4_cuda_kernel_seed2026.json
```

## Artifacts

- Python wrapper: `neural_engine/qwen_correction_dispatch.py`;
- C++ binding: `neural_engine/qwen_correction_dispatch.cpp`;
- CUDA kernel: `neural_engine/qwen_correction_dispatch.cu`;
- A/B harness: `benchmark_qwen_trained_graph_audit.py`;
- quality/runtime candidate: `RUNTIME_QWEN_CORRECTION_RANK4_AUDIT_20260909.md`.
