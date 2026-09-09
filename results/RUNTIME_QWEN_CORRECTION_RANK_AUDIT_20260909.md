# V0.204 — Trained correction-rank audit

**Date:** 2026-09-09  
**Status:** `RANK-32 ACCEPTED OPT-IN; DEFAULT 64 PRESERVED`  
**Branch:** `exp/track-runtime`

## Question

The rank-64 cross-group correction preserved quality but remained expensive at
larger decode batches. This audit tests rank 32 and rank 16 under the same
trained K=5 recipe. Routing, eight replaced layers, hard route scale, training
steps, and evaluation data remain unchanged.

## Quality and runtime results

| rank | seed | CE delta | B8 graph / dense | B8 graph / sparse eager | max graph parity |
|---:|---:|---:|---:|---:|---:|
| 64 | 2026 | `+0.035745` | `1.491x` | — | `<1e-5` |
| 32 | 2026 | `+0.031617` | `1.411x` | `1.017x` | `1.00e-5` |
| 32 | 17 | `+0.039860` | `1.344x` | `1.031x` | `9.54e-6` |
| 16 | 2026 | `+0.036007` | `1.479x` | `1.051x` | `9.54e-6` |

The quality gate is `CE delta < +0.05`; all three tested ranks pass it on the
shown seeds. Rank 32 is the best practical compromise in this small audit:
it reduces B8 graph/dense overhead relative to rank 64 and remains stable on a
second seed. Rank 16 does not improve on rank 32, so further rank reduction is
not accepted as a general strategy.

This is not a large architecture breakthrough. At B8 the sparse graph is still
slower than the dense parent; the remaining bottleneck is correction and
dispatch overhead. Rank 32 is therefore opt-in only until a larger-seed and
longer-run audit confirms it.

## Decision

- keep rank 64 as the compatibility/default recipe;
- retain rank 32 as the current lower-rank opt-in candidate;
- reject rank 16 as the preferred runtime setting;
- continue toward a fused/static-index correction kernel rather than forcing
  lower rank.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 32 --iterations 30 --correction-backend-iterations 20 --output results/runs/v0_204_rank32.json
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 16 --iterations 30 --correction-backend-iterations 20 --output results/runs/v0_204_rank16.json
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 32 --seed 17 --iterations 20 --correction-backend-iterations 15 --batch-sizes 1 8 --output results/runs/v0_204_rank32_seed17.json
```

## Artifacts

- parameterized audit: `benchmark_qwen_trained_graph_audit.py`;
- correction implementation and BMM path: `benchmark_qwen_multi_layer_transplant.py`;
- prior rank-64 BMM control: `results/RUNTIME_QWEN_TRAINED_CORRECTION_BMM_AUDIT_20260909.md`.
