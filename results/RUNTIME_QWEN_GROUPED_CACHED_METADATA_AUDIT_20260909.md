# V0.231 — Grouped route-metadata cache audit

**Date:** 2026-09-09  
**Status:** `PARITY-SAFE MICRO-OPTIMIZATION; NOT A SPEED BREAKTHROUGH`  
**Branch:** `exp/track-runtime`

## Question

The grouped selected-FFN path rebuilt route-independent `arange`, token-index
and slot-index tensors on every layer and decode step. This opt-in variant
caches those tensors by `(token_count, active_experts, device)` for fixed
shapes. Routing, selected experts, projections, correction and model defaults
are unchanged.

## Protocol

The trained Qwen K=5 recipe is unchanged: Qwen3-0.6B layers 19–26, `E=8`,
`K=5`, subset-router, correction rank 1, float32, and `600/600/200` child,
hard and router steps. Prefix length is 4. Each seed uses 8 CUDA-Graph
warmups and 20 timing iterations at B1/B8/B32. The cached path is compared
directly with ordinary grouped and dense parent timing.

## Results

`cached/grouped` graph-time ratios at B1/B8/B32:

| Seed | Quality CE delta | B1 | B8 | B32 | Generation parity |
|---:|---:|---:|---:|---:|:---:|
| 2026 | `+0.045874` | `0.996x` | `1.000x` | `1.002x` | exact |
| 17 | `+0.037267` | `0.997x` | `0.994x` | `0.996x` | exact |

Relative to dense, cached grouped graph time is `1.078x/1.046x/1.074x`
for seed2026 and `1.075x/1.044x/1.066x` for seed17 at B1/B8/B32.
All route outputs, final logits and eight-token graph/eager generations passed
the existing parity checks. The small timing movement is within the expected
GPU measurement noise and is not a dense-serving win.

## Decision

- Keep the metadata cache as an opt-in `dispatch_mode="grouped-cached"`
  experiment; it is numerically safe and has no material memory cost.
- Do not make it the default or claim a meaningful speedup.
- Mark the route-metadata hypothesis as exhausted for the next optimization
  cycle. The remaining gap is in selected FFN packing/projection/scatter
  execution, which requires a tiled or fully fused kernel rather than more
  Python-side metadata reuse.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --experiment V0.231_grouped_cached_metadata_audit --include-cached-grouped --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --output results/runs/v0_231_grouped_cached_metadata_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --experiment V0.231_grouped_cached_metadata_audit --include-cached-grouped --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --output results/runs/v0_231_grouped_cached_metadata_seed17.json
```

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `tests/test_qwen_packed_dispatch.py`
- `results/runs/v0_231_grouped_cached_metadata_seed2026.json`
- `results/runs/v0_231_grouped_cached_metadata_seed17.json`
