# V0.232 — Grouped selected-output/correction fusion audit

**Date:** 2026-09-09  
**Status:** `ACCEPTED OPT-IN MICRO-OPTIMIZATION; DENSE GAP OPEN`  
**Branch:** `exp/track-runtime`

## Question

The grouped path already computes selected FFN outputs in expert-major order,
then reorders them to `[token, K, hidden]` so the wrapper can run the low-rank
cross-group correction in a second pass. This opt-in variant applies the
correction while the outputs are still in expert-major order and accumulates
the corrected selected outputs directly. It leaves routing, projection
weights, correction formula and model defaults unchanged.

## Protocol

The trained Qwen K=5 recipe is unchanged: Qwen3-0.6B layers 19–26, `E=8`,
`K=5`, subset-router, float32, correction rank 1, and `600/600/200` child,
hard and router steps. Prefix length is 4. Each seed uses 8 CUDA-Graph
warmups and 20 timing iterations at B1/B8/B32. The fused correction path is
compared with ordinary grouped, grouped-fused and the dense parent.

## Results

| Seed | Quality CE delta | B1 fused/grouped | B8 fused/grouped | B32 fused/grouped | B1 fused/dense | B8 fused/dense | B32 fused/dense | Generation |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2026 | `+0.045226` | `0.997x` | `0.988x` | `0.992x` | `1.081x` | `1.039x` | `1.066x` | exact |
| 17 | `+0.040347` | `0.991x` | `0.989x` | `0.993x` | `1.072x` | `1.037x` | `1.061x` | exact |

The quality gate is `CE delta <= +0.05`; both seeds pass. The fused path is
faster than ordinary grouped in all six shape/seed comparisons, with a mean
improvement of about `0.8%`. This is a consistent but small optimization; the
remaining dense gap is about `3.7–8.1%`.

All graph/eager parity checks passed. The grouped and grouped-correction-fused
eight-token greedy generations matched exactly on both seeds, and maximum
final-logit differences versus the single-token reference stayed below
`1.5e-5`.

## Decision

- Keep the correction fusion as an opt-in backend
  (`correction_dispatch_backend="grouped-fused-correction"`).
- Do not change the default or claim that sparse serving is faster than dense.
- The fusion validates that the wrapper-side reorder/correction pass has a
  measurable contribution, but its ceiling is small. The next major runtime
  target is still a tiled/static-index or full selected-FFN kernel that fuses
  packing, projection and accumulation more deeply.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --experiment V0.232_grouped_correction_fused_audit --include-grouped-correction-fused --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --output results/runs/v0_232_grouped_correction_fused_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --experiment V0.232_grouped_correction_fused_audit --include-grouped-correction-fused --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --output results/runs/v0_232_grouped_correction_fused_seed17.json
```

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `tests/test_qwen_packed_dispatch.py`
- `results/runs/v0_232_grouped_correction_fused_seed2026.json`
- `results/runs/v0_232_grouped_correction_fused_seed17.json`
