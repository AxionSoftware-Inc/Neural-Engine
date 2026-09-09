# V0.215 — Fused correction batch sweep

**Date:** 2026-09-09  
**Status:** `OPEN; NO AUTOMATIC BATCH POLICY`  
**Branch:** `exp/track-runtime`

## Question

V0.214 showed a positive B8 result for the fused base-output plus correction
kernel. This sweep checks whether that advantage also holds for one-token
batch-1 decode, using the same trained K=5/rank-4 recipe and the same graph
backend matrix.

## Results

The table reports graph milliseconds for the vectorized and fused-full
correction backends, followed by fused-full/vectorized. The model and route
are otherwise unchanged.

| seed | batch | vectorized | fused-full | fused/vectorized | max fused vs vectorized error |
|---:|---:|---:|---:|---:|---:|
| 42 | 1 | `11.175 ms` | `14.256 ms` | `1.276x` | `1.05e-5` |
| 42 | 8 | `28.634 ms` | `24.672 ms` | `0.862x` | `1.53e-5` |
| 2026 | 1 | `21.715 ms` | `15.631 ms` | `0.720x` | `9.54e-6` |
| 2026 | 8 | `52.617 ms` | `26.068 ms` | `0.495x` | `1.29e-5` |

All four fused cases pass graph/eager parity, and the full-logit differences
against vectorized correction stay below `1.6e-5`. B8 is faster in both seeds.
Batch-1 timing is not stable enough across these two runs to justify an
automatic switch: one seed favors vectorized correction and the other favors
fused-full. The CUDA timing spread also shows that a repeated median benchmark
is needed before making a fine-grained batch policy.

## Decision

- Keep `cuda-fused-full` as an opt-in backend for controlled B8/large-batch
  experiments.
- Keep vectorized correction as the default, especially for batch-1 decode.
- Do not add a runtime batch-size switch from this sweep alone.
- The next runtime experiment should use repeated interleaved timing and/or
  fuse router/top-k with dispatch; the current kernel does not remove those
  costs, attention, or dynamic-shape constraints.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --child-steps 300 --hard-steps 300 --router-steps 100 --seed 42 --warmup 40 --iterations 100 --correction-backend-iterations 15 --single-token-backend-iterations 30 --compiled-child-iterations 2 --batch-sizes 1 8 --output results/runs/v0_215_full_correction_batch_sweep_seed42.json
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --child-steps 300 --hard-steps 300 --router-steps 100 --seed 2026 --warmup 40 --iterations 100 --correction-backend-iterations 15 --single-token-backend-iterations 30 --compiled-child-iterations 2 --batch-sizes 1 8 --output results/runs/v0_215_full_correction_batch_sweep_seed2026.json
```

## Validation

The repository regression suite remains `158 passed, 2 warnings`; the packed
backend is still intentionally probed last and still fails graph capture for
its known device-side `torch.where` path.
