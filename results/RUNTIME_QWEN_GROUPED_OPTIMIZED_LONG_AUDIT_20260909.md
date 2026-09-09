# V0.237 — Long repeated combined grouped-path audit

**Date:** 2026-09-09  
**Status:** `BEST CURRENT OPT-IN PATH; NOT DEFAULT`  
**Branch:** `exp/track-runtime`

## Purpose

The short V0.236 sweep showed a large seed-to-seed timing difference. This
follow-up repeats the combined `grouped-optimized` path with a longer timing
window to separate a real runtime effect from GPU timing noise.

The path combines route-independent metadata caching, prepacked BMM operands,
fused gate/value projection, grouped selected-output/correction fusion, and
the exact uniform K=5 accumulation shortcut. It does not change model weights,
routing, or quality training.

## Protocol

Qwen3-0.6B, layers 19–26, `E=8`, `K=5`, rank 1, float32, and
`600/600/200` child/hard/router steps. Prefix length is 4. Each seed uses 20
CUDA-Graph warmups and 50 timing iterations at B8/B32. The dense parent and
existing grouped path are measured in the same process. Eight-token
fixed-shape generation and full-model graph/eager parity are checked.

## Results

Graph time relative to the dense parent, in B8/B32 order:

| Seed | Quality CE delta | Existing grouped | Combined optimized | Optimized / grouped | Generation |
|---:|---:|:---|:---|:---|:---:|
| 2026 | `+0.046924` | `1.046x / 1.070x` | `1.010x / 1.050x` | `0.965x / 0.981x` | exact |
| 17 | `+0.040464` | `1.048x / 1.073x` | `1.018x / 1.058x` | `0.972x / 0.986x` | exact |

Thus the combined path has a repeatable small gain over grouped: about
`2.8–3.5%` at B8 and `1.4–1.8%` at B32. The earlier V0.236 B32 result of
`0.995x` dense for seed2026 and `1.118x` for seed17 was not a stable estimate;
the longer repeat puts the seeds at `1.050x` and `1.058x` dense instead.

The quality deltas remain inside the current `+0.05` gate, but the runtime
mode itself provides no quality gain. All generation comparisons are exact;
maximum graph/eager logit differences remain around the existing float32
audit range (`1.2e-5`).

## Decision

Keep `grouped-optimized` as the best current opt-in runtime configuration for
medium batches. Do not make it the global default yet: it is still about
`1.01–1.06x` dense at B8/B32 and the B1 result remains materially worse than
dense in the earlier shape sweep. Its benefit comes from several small
operator-level reductions, not from solving selected-FFN dispatch.

The next meaningful target is route-count-aware packing plus an autotuned
grouped-GEMM backend, or a kernel that fuses route metadata, packing,
projection, correction, and accumulation. More isolated Python shortcuts are
unlikely to close the remaining B1 and small-batch gap.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 20 --iterations 50 --batch-sizes 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --experiment V0.237_grouped_optimized_repeat_seed2026 --include-grouped-optimized --output results/runs/v0_237_grouped_optimized_repeat_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 20 --iterations 50 --batch-sizes 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --experiment V0.237_grouped_optimized_repeat_seed17 --include-grouped-optimized --output results/runs/v0_237_grouped_optimized_repeat_seed17.json
```

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `results/runs/v0_237_grouped_optimized_repeat_seed2026.json`
- `results/runs/v0_237_grouped_optimized_repeat_seed17.json`
