# V0.236 — Combined grouped runtime path audit

**Date:** 2026-09-09  
**Status:** `CONDITIONAL OPT-IN; NOT A STABLE DEFAULT`  
**Branch:** `exp/track-runtime`

## Question

Several small runtime probes were individually safe: route-independent
metadata caching, contiguous BMM weight operands, fused gate/value projection,
selected-output/correction fusion, and the exact uniform K=5 accumulation
shortcut. This experiment checks whether combining them produces a stable
end-to-end serving gain.

The new `grouped-optimized` mode combines the first three, while the benchmark
also enables correction fusion and uniform accumulation. The model, router,
training recipe, and dense reference are unchanged.

## Protocol

Qwen3-0.6B, layers 19–26, `E=8`, `K=5`, rank 1, float32, and
`600/600/200` child/hard/router steps. Prefix length is 4. Each seed uses 8
CUDA-Graph warmups and 20 timing iterations at B1/B8/B32. The same run checks
hard-route numerical parity and eight-token fixed-shape generation.

## Results

Graph time relative to the dense parent, in B1/B8/B32 order:

| Seed | Quality CE delta | Existing grouped | Combined optimized | Optimized / grouped | Generation |
|---:|---:|:---|:---|:---|:---:|
| 2026 | `+0.046377` | `1.165x / 1.114x / 1.076x` | `1.147x / 1.078x / 0.995x` | `0.985x / 0.968x / 0.924x` | exact |
| 17 | `+0.037298` | `1.089x / 1.048x / 1.073x` | `1.157x / 1.084x / 1.118x` | `1.062x / 1.035x / 1.042x` | exact |

The combined path is faster for seed2026, including a small B32 win over
dense, but slower for seed17 at every tested batch size. This is not a stable
serving result; route-dependent packed shapes and GPU scheduling still move
the measured balance. Maximum full-model graph/eager logit differences remain
in the existing float32 audit range (about `1.5e-5`), and all generation token
comparisons are exact.

The quality values are unchanged by the runtime mode. Both seeds remain inside
the current `+0.05` CE-delta gate, but no quality improvement is claimed.

## Decision

Keep `grouped-optimized` as an opt-in diagnostic and do not make it the
default. It demonstrates that the small optimizations can interact
non-additively: they do not reliably close the sparse/dense gap across seeds.

The next meaningful runtime work must remove route-dependent packing and
launch overhead itself, using either a shape-aware autotuned grouped-GEMM
backend or a kernel that fuses route metadata, packing, selected projection,
correction, and accumulation. More isolated Python shortcuts are unlikely to
produce a dependable large gain.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --experiment V0.236_grouped_optimized_combo_audit --include-grouped-optimized --output results/runs/v0_236_grouped_optimized_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --experiment V0.236_grouped_optimized_combo_audit --include-grouped-optimized --output results/runs/v0_236_grouped_optimized_seed17.json
```

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `tests/test_qwen_packed_dispatch.py`
- `results/runs/v0_236_grouped_optimized_seed2026.json`
- `results/runs/v0_236_grouped_optimized_seed17.json`
