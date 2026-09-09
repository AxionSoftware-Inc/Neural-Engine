# V0.240 — Grouped selected-FFN stage profile

**Date:** 2026-09-09  
**Status:** `BOTTLENECK IDENTIFIED; NO DEFAULT CHANGE`  
**Branch:** `exp/track-runtime`

## Purpose

The V0.238–V0.239 shape-aware policy reduced the grouped path's runtime
overhead, but it did not explain which internal operation still separates the
sparse path from dense execution. This audit adds profiler ranges around the
five major grouped selected-FFN stages:

1. route metadata construction;
2. expert-major packing;
3. selected expert projections;
4. selected-output lookup and correction;
5. token accumulation.

The ranges are diagnostic only. They do not change route selection, model
weights, correction rank, active K, or the model's output math.

## Protocol

The run used the local `Qwen/Qwen3-0.6B` checkpoint, layers `19–26`,
`E=8`, `K=5`, rank-1 correction, float32 CUDA, the trained cascade recipe
`600/600/200`, and the current `grouped-adaptive` backend with grouped
correction fusion and uniform K=5 accumulation. A four-token KV prefix was
used. Each shape had three warmups and five profiler iterations; the profiler
recorded eight routed layers per forward, or 40 calls per stage.

## Result

Values below are profiler `total CUDA` time across all eight layers and five
profile iterations. Percentages are within the five instrumented ranges only.

| batch | metadata | pack | projections | select/correction | accumulate |
|---:|---:|---:|---:|---:|---:|
| 1 | `7.021 ms` / 6.4% | `50.519 ms` / **46.2%** | `14.027 ms` / 12.8% | `32.140 ms` / **29.4%** | `5.735 ms` / 5.2% |
| 8 | `1.447 ms` / 1.5% | `47.441 ms` / **49.0%** | `11.014 ms` / 11.4% | `30.950 ms` / **32.0%** | `5.870 ms` / 6.1% |
| 32 | `1.652 ms` / 1.6% | `49.636 ms` / **48.7%** | `18.789 ms` / 18.4% | `25.970 ms` / **25.5%** | `5.877 ms` / 5.8% |

The stage order is stable: `pack` is the largest cost at every tested batch,
followed by selected-output lookup/correction. Projection GEMMs are not the
dominant cost at B1 or B8. Metadata construction becomes small after cached
metadata is active, so the remaining overhead is not explained by rebuilding
`arange` indices alone.

## Decision

- The current evidence does **not** support a router-only explanation for the
  remaining runtime gap.
- More independent Python-level cache flags are unlikely to produce a large
  gain: the dominant `pack` and `select/correction` work remains in the
  forward path.
- Keep `grouped-adaptive` as the best opt-in policy; do not change the global
  default from this diagnostic.
- The next runtime experiment should prototype a route-count-aware selected
  dispatch path that combines packing, selected output extraction, and
  accumulation, ideally as one compiled CUDA/Triton kernel. The existing
  single-token gather path remains the correctness/runtime control, not a new
  quality claim.

This is a systems result only. It does not improve CE, routing regret, or
model quality, and it does not imply that the attention-free architecture's
capacity problem has been solved.

## Reproduction

```text
python -u benchmark_qwen_grouped_stage_profile.py --batch-sizes 1 8 32 --prefix-lengths 4 --profile-iterations 5 --warmup 3 --child-steps 600 --hard-steps 600 --router-steps 200 --calibration-rank 1 --seed 2026 --output results/runs/v0_240_grouped_stage_profile_seed2026.json
```

## Artifacts

- `benchmark_qwen_grouped_stage_profile.py`
- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/v0_240_grouped_stage_profile_seed2026.json`
- `tests/test_qwen_packed_dispatch.py`
