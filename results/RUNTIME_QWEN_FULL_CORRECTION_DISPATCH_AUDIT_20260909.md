# V0.214 — Fused base-output plus correction dispatch

**Date:** 2026-09-09  
**Status:** `ACCEPTED OPT-IN; DEFAULT UNCHANGED`  
**Branch:** `exp/track-runtime`

## Hypothesis

The existing CUDA correction kernel removes only the separate low-rank
correction launch. The selected Qwen SwiGLU output is still computed by the
normal single-token path, so the sparse child retains multiple projection and
gather operations. A fixed-shape kernel that computes the selected group
output and its low-rank correction in one dispatch may reduce this overhead.

The new path keeps the same route IDs, route weights, group weights and
correction weights. It is float32, CUDA-only, one-token only, and opt-in.
The normal vectorized backend remains the default and is the reference.

## Protocol

- model: local `Qwen/Qwen3-0.6B`, layers 19–26;
- sparse child: E=8, K=5, contiguous groups, trained cross-group correction;
- correction rank: 4;
- same child/router training recipe for both arms: `300/300/100` steps;
- B8 fixed-KV graph benchmark, 40 warmups and 100 timed iterations;
- comparison: vectorized correction versus `cuda-fused-full`;
- quality and route state are unchanged; only the inference dispatch backend
  changes.

## Results

| seed | CE delta | vectorized graph | fused-full graph | fused/vectorized | dense parent | fused/parent | full parity vs vectorized | graph/eager parity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | `+0.040382` | `30.546 ms` | `25.735 ms` | `0.842x` | `26.900 ms` | `0.957x` | `1.62e-5` | `1.34e-5` |
| 2026 | `+0.033775` | `28.649 ms` | `24.608 ms` | `0.859x` | `24.620 ms` | `1.000x` | `1.67e-5` | `1.34e-5` |

Both seeds pass the existing `CE delta < +0.05` quality gate. The fused path
is about 14–16% faster than vectorized correction in the same B8 backend A/B,
and its final-logit error remains far below the `1e-3` graph tolerance. The
full sparse graph is approximately dense-parent speed, but this is not yet a
large end-to-end speedup over dense Qwen.

## Decision

- **Accept as opt-in:** `correction_dispatch_backend="cuda-fused-full"`.
- Keep `correction_dispatch_backend="vectorized"` as the default until a
  wider shape/dtype/stream audit is complete.
- Do not claim that this solves the whole one-token serving problem: router,
  top-k, attention, cache management and kernel launch overhead remain.
- Do not change model quality or routing defaults based on this runtime-only
  result.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --child-steps 300 --hard-steps 300 --router-steps 100 --seed 42 --warmup 40 --iterations 100 --correction-backend-iterations 15 --single-token-backend-iterations 30 --compiled-child-iterations 2 --batch-sizes 1 8 --output results/runs/v0_214_full_correction_seed42.json
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --child-steps 300 --hard-steps 300 --router-steps 100 --seed 2026 --warmup 40 --iterations 100 --correction-backend-iterations 15 --single-token-backend-iterations 30 --compiled-child-iterations 2 --batch-sizes 1 8 --output results/runs/v0_214_full_correction_seed2026.json
```

## Artifacts

- Python wrapper: `neural_engine/qwen_full_correction_dispatch.py`;
- C++ binding: `neural_engine/qwen_full_correction_dispatch.cpp`;
- CUDA kernel: `neural_engine/qwen_full_correction_dispatch.cu`;
- integrated benchmark: `benchmark_qwen_trained_graph_audit.py`;
- full regression: `158 passed, 2 warnings`.
