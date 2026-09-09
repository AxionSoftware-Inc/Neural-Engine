# V0.210 — Rank-4 long-budget audit

**Date:** 2026-09-09  
**Status:** `PASS; RANK-4 OPT-IN RETAINED; DEFAULT 64 PRESERVED`  
**Branch:** `exp/track-runtime`

## Question

The three-seed rank-4 audit used 300 child steps, 300 hard-routing steps, and
100 router steps. This audit doubles those training budgets to test whether the
rank-4 quality result is a short-budget artifact. The architecture, rank,
router, eight replaced layers, data, and evaluation protocol remain unchanged.

## Results

| seed | child/hard/router steps | CE delta | top-1 agreement | B8 graph / dense | B8 graph / sparse eager | max parity | generation |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2026 | `600/600/200` | `+0.027947` | `0.7937` | `1.280x` | `0.904x` | `9.54e-6` | exact |
| 17 | `600/600/200` | `+0.038252` | `0.7944` | `1.265x` | `0.873x` | `9.06e-6` | exact |

The quality gate is `CE delta < +0.05`; both seeds pass. Compared with the
short-budget rank-4 runs, the CE delta changes from `+0.037330` to `+0.027947`
on seed 2026 and from `+0.040037` to `+0.038252` on seed 17. This is not a
quality regression. Graph replay remains well below the existing `1e-3`
tolerance, and both graph/eager generation and reused-shape generation match
exactly.

## Decision

- retain rank 4 as the smallest tested viable correction opt-in;
- do not promote rank 4 to the default yet; rank 64 remains the compatibility
  default;
- treat the short-budget artifact hypothesis as weakened by this two-seed
  long-budget pass;
- continue toward a fused/static-index correction kernel, since B8 graph is
  still `1.265x–1.280x` the dense parent.

The better quality under the longer budget is useful evidence about training,
not a claim that rank 4 solves the architectural routing gap. The next quality
check can use rank 4 as the fixed candidate; the next performance check should
target correction dispatch rather than further reducing rank blindly.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --warmup 40 --iterations 100 --correction-backend-iterations 15 --single-token-backend-iterations 30 --compiled-child-iterations 2 --batch-sizes 1 8 --output results/runs/v0_210_rank4_long_seed2026.json
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --warmup 40 --iterations 100 --correction-backend-iterations 15 --single-token-backend-iterations 30 --compiled-child-iterations 2 --batch-sizes 1 8 --output results/runs/v0_210_rank4_long_seed17.json
```

## Artifacts

- parameterized audit: `benchmark_qwen_trained_graph_audit.py`;
- rank-4 short-budget audit: `RUNTIME_QWEN_CORRECTION_RANK4_AUDIT_20260909.md`;
- correction implementation: `benchmark_qwen_multi_layer_transplant.py`;
- no-correction control: `RUNTIME_QWEN_CORRECTION_ABLATION_20260909.md`.
