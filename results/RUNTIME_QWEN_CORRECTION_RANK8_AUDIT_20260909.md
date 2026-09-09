# V0.205 — Rank-8 correction audit

**Date:** 2026-09-09  
**Status:** `RANK-8 ACCEPTED OPT-IN; DEFAULT 64 PRESERVED`  
**Branch:** `exp/track-runtime`

## Result

Rank 8 was tested with the same trained K=5 recipe, eight replaced layers,
and the same graph-safe BMM correction path. The primary timing uses 40 warmup
and 100 measurement iterations to reduce short-run timing noise.

| seed | CE delta | B8 parent | B8 sparse eager | B8 graph | graph / dense | graph / sparse eager | max parity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | `+0.024122` | `25.650 ms` | `38.603 ms` | `35.665 ms` | `1.390x` | `0.924x` | `9.54e-6` |
| 17 | `+0.029971` | `34.711 ms` | `36.263 ms` | `34.988 ms` | `1.008x` | `0.965x` | `9.54e-6` |

Both runs pass the quality gate (`CE delta < +0.05`). Greedy generation also
matched eager output exactly and graph shape reuse passed. Rank 8 reduces the
trained correction overhead substantially compared with the rank-64 control
while preserving output parity.

Absolute latency varied with GPU state: short 20-iteration runs gave wider
ratios. The longer runs above are the decision measurements, and both remain
better than the rank-64 BMM control (`1.491x` at B8 in V0.203).

## Decision

- retain rank 8 as the current fastest opt-in correction configuration;
- keep rank 64 as the default compatibility recipe;
- do not claim a quality breakthrough — the gain is runtime/parameter
  efficiency, not a new routing solution;
- continue testing a fused/static-index correction kernel before changing the
  default.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 8 --seed 2026 --warmup 40 --iterations 100 --correction-backend-iterations 30 --batch-sizes 1 8 --output results/runs/v0_205_rank8_seed2026_long.json
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 8 --seed 17 --warmup 40 --iterations 100 --correction-backend-iterations 30 --batch-sizes 1 8 --output results/runs/v0_205_rank8_seed17_long.json
```

## Artifact

- parameterized audit: `benchmark_qwen_trained_graph_audit.py`;
- correction implementation: `benchmark_qwen_multi_layer_transplant.py`;
- rank-32/rank-16 control: `results/RUNTIME_QWEN_CORRECTION_RANK_AUDIT_20260909.md`.
