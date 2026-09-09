# V0.217 — Interleaved fused-correction timing

**Date:** 2026-09-09  
**Status:** `SPEED HYPOTHESIS REJECTED; PARITY-SAFE OPT-IN RETAINED`  
**Branch:** `exp/track-runtime`

## Purpose

Separate backend timing was unstable: the fused-full kernel had appeared
14–16% faster than vectorized correction in one ordered A/B, but a repeated
median run did not reproduce that gain. This audit captures both graphs and
alternates them on the same CUDA stream, synchronizing after each replay so
both arms share the same clock and thermal window.

## Results

| seed | batch | vectorized graph | fused-full graph | fused/vectorized | max final-logit error |
|---:|---:|---:|---:|---:|---:|
| 42 | 1 | `17.043 ms` | `25.195 ms` | `1.478x` | `8.58e-6` |
| 42 | 8 | `37.982 ms` | `37.139 ms` | `0.978x` | `1.34e-5` |
| 2026 | 1 | `17.253 ms` | `25.334 ms` | `1.468x` | `1.00e-5` |
| 2026 | 8 | `38.967 ms` | `37.979 ms` | `0.975x` | `1.53e-5` |

The fused path is consistently slower for batch-1 decode and only about 2–3%
faster at B8 under the interleaved measurement. All four cases preserve
graph output parity well below the `1e-3` tolerance. The earlier 14–16% A/B
gain is therefore not a reliable performance claim.

## Decision

- Reject `cuda-fused-full` as a general speed solution and do not make it the
  default.
- Retain the implementation as a parity-safe opt-in for future fused-kernel
  work and controlled experiments.
- Do not add an automatic batch-size policy from these measurements.
- The next meaningful runtime target is a kernel that fuses router score,
  top-k/index materialization, selected-group dispatch and correction; merely
  combining the final base output with correction is not enough.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --child-steps 300 --hard-steps 300 --router-steps 100 --seed 42 --warmup 20 --iterations 60 --correction-backend-iterations 10 --single-token-backend-iterations 20 --compiled-child-iterations 1 --interleaved-timing-iterations 30 --batch-sizes 1 8 --output results/runs/v0_217_fused_correction_interleaved_seed42.json
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --child-steps 300 --hard-steps 300 --router-steps 100 --seed 2026 --warmup 20 --iterations 60 --correction-backend-iterations 10 --single-token-backend-iterations 20 --compiled-child-iterations 1 --interleaved-timing-iterations 30 --batch-sizes 1 8 --output results/runs/v0_217_fused_correction_interleaved_seed2026.json
```

## Validation

The final valid runs report `PARITY_PASS`; the packed capture failure remains
the intentionally last probe. Repository tests remain `158 passed, 2 warnings`.
