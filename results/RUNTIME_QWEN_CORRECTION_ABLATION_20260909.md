# V0.208 — No-correction ablation

**Date:** 2026-09-09  
**Status:** `RANK-0 REJECTED; CORRECTION REQUIRED`  
**Branch:** `exp/track-runtime`

## Question

Rank 8 improved runtime while preserving the quality gate. This ablation
removes the cross-group correction entirely (`calibration_rank=0`) while
keeping the same E=8/K=5 router, eight replaced layers, training data, and
router training. It tests whether correction is actually necessary.

## Result

| metric | rank 0 |
|---|---:|
| teacher CE | `4.785308` |
| sparse CE | `4.876332` |
| CE delta | `+0.091023` |
| top-1 agreement | `0.7788` |
| B8 parent | `31.066 ms` |
| B8 sparse graph | `32.526 ms` |
| B8 graph / dense | `1.047x` |
| B8 graph / sparse eager | `0.898x` |
| max graph parity error | `9.06e-6` |

The runtime is close to dense at B8, but the quality gate is clearly failed
(`+0.091023 > +0.05`). Greedy generation and graph replay parity still pass;
the failure is representational quality, not execution correctness.

## Decision

- reject rank 0 as a quality configuration;
- keep a low-rank correction rather than removing correction capacity;
- retain rank 8 as the current quality/runtime opt-in candidate;
- keep rank 64 as the default compatibility recipe until longer validation;
- continue toward a fused correction implementation, because removing the
  correction is not an acceptable speed solution.

The audit skips child-training when rank 0 leaves no trainable correction
parameters. This fixes a benchmark-only `loss.backward()` failure and does not
change the rank-0 model result.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 0 --seed 2026 --warmup 40 --iterations 100 --correction-backend-iterations 15 --single-token-backend-iterations 30 --compiled-child-iterations 2 --batch-sizes 1 8 --output results/runs/v0_208_rank0_seed2026.json
```

## Artifact

- rank parameter and ablation harness: `benchmark_qwen_trained_graph_audit.py`;
- correction implementation: `benchmark_qwen_multi_layer_transplant.py`;
- rank-8 control: `results/RUNTIME_QWEN_CORRECTION_RANK8_AUDIT_20260909.md`.
