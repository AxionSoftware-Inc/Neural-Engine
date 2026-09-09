# V0.233 — Uniform K-subset accumulation audit

**Date:** 2026-09-09  
**Status:** `REJECTED FOR ADOPTION; OPT-IN DIAGNOSTIC ONLY`  
**Branch:** `exp/track-runtime`

## Question

The accepted hard K=5 subset-router gives identical logits to each member of
the selected subset, so the route softmax is uniform (`1/K`) and
`hard_route_scale=K` cancels it. This experiment removes the per-pair route
weight multiply and uses a single accumulation scale. It is enabled only for
the subset-router and only through an explicit opt-in benchmark flag.

## Protocol

The trained Qwen K=5 recipe is unchanged: Qwen3-0.6B layers 19–26, `E=8`,
`K=5`, rank 1, float32, grouped selected-FFN dispatch, and `600/600/200`
child/hard/router steps. Prefix length is 4. Each seed uses 8 CUDA-Graph
warmups and 20 timing iterations at B1/B8/B32. The uniform path is compared
with the grouped correction-fused path.

## Results

`uniform/correction-fused` graph-time ratios at B1/B8/B32:

| Seed | Quality CE delta | B1 | B8 | B32 | Generation parity |
|---:|---:|---:|---:|---:|:---:|
| 2026 | `+0.046322` | `0.996x` | `0.998x` | `1.000x` | exact |
| 17 | `+0.039829` | `1.002x` | `0.996x` | `0.995x` | exact |

Relative to the dense parent, the uniform path is `1.076x/1.030x/1.066x`
for seed2026 and `1.072x/1.038x/1.057x` for seed17 at B1/B8/B32. The
quality gate remains inside `+0.05`, and all graph/eager and eight-token
generation parity checks pass.

The timing gain is small and inconsistent: the first seed is effectively
neutral at B32, while the second seed is slightly slower at B1. It does not
justify a serving policy or a default change.

## Decision

- Reject this as a material speed solution.
- Keep the implementation as an opt-in diagnostic for the exact uniform
  subset-router contract; do not apply it to ordinary routers with non-uniform
  weights.
- Keep grouped correction fusion as the better opt-in micro-optimization.
- The main runtime problem remains selected-FFN packing/projection/scatter;
  the next major experiment must target a tiled or fully fused kernel.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --experiment V0.233_grouped_uniform_accum_audit --include-grouped-correction-fused --include-grouped-uniform-fused --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --output results/runs/v0_233_grouped_uniform_accum_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --experiment V0.233_grouped_uniform_accum_audit --include-grouped-correction-fused --include-grouped-uniform-fused --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --output results/runs/v0_233_grouped_uniform_accum_seed17.json
```

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `tests/test_qwen_packed_dispatch.py`
- `results/runs/v0_233_grouped_uniform_accum_seed2026.json`
- `results/runs/v0_233_grouped_uniform_accum_seed17.json`
