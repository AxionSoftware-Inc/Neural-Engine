# V0.224 — Grouped-fused selected-FFN audit

**Date:** 2026-09-09  
**Status:** `PARITY-SAFE MICRO-OPTIMIZATION; NOT A DENSE-SERVING WIN`  
**Branch:** `exp/track-runtime`

## Question

V0.223 made grouped selected-FFN dispatch CUDA-Graph-safe and showed a large
improvement over the single-token path. This follow-up tests the existing
`grouped-fused` variant, which concatenates each group's gate and value
projection into one BMM. The model is an eight-layer trained K=5
subset-router child; only the dispatch path changes.

## Results

Seed 2026, accepted `300/300/100`, rank-64 recipe, 8 warmups and 10 timing
iterations. `grouped-fused/grouped` shows the incremental effect of combining
the two projections:

| Batch | Grouped eager | Grouped-fused eager | Fused/grouped | Grouped graph | Grouped-fused graph | Fused/grouped |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.0274 ms | 40.1403 ms | `0.912x` | 11.5409 ms | 11.5374 ms | `1.000x` |
| 8 | 37.4574 ms | 35.0215 ms | `0.935x` | 19.0465 ms | 18.9160 ms | `0.993x` |
| 32 | 48.2048 ms | 48.4267 ms | `1.005x` | 41.2609 ms | 41.1226 ms | `0.997x` |

The quality gate was unchanged and passed: teacher CE `4.785308`, sparse CE
`4.825112`, CE delta `+0.039804`. The maximum eager logit difference versus
the single-token reference was `7.63e-6/9.54e-6/1.34e-5` for the
grouped-fused B1/B8/B32 probes. Eight-token CUDA-Graph generation matched both
the single-token and grouped paths exactly.

The dense parent comparison remains the limiting result. Grouped-fused sparse
graph divided by dense parent graph was `1.108x/1.136x/1.257x` at B1/B8/B32.
Therefore the sparse path is still slower end-to-end even though it is much
faster than the previous sparse single-token implementation.

## Decision

- Keep `grouped-fused` as a parity-safe opt-in variant.
- The graph benefit over ordinary grouped is only about `0.0–0.7%`; it is not
  a new large optimization.
- At eager B1/B8 it helps, but B32 is neutral/slightly slower, so no automatic
  batch policy is added.
- The dense-serving gap remains open. The next meaningful runtime work must
  reduce selected FFN launch/packing overhead or fuse more of the layer; more
  router-only work is not justified by these results.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 8 --iterations 10 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 64 --seed 2026 --output results/runs/v0_224_trained_grouped_fused_generation.json
```

## Artifacts

- `benchmark_qwen_trained_dispatch_path_audit.py`
- graph-safe grouped change in `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/v0_224_trained_grouped_fused_audit.json`
- `results/runs/v0_224_trained_grouped_fused_generation.json`
