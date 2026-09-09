# V0.234 — Grouped prepacked-weight layout audit

**Date:** 2026-09-09  
**Status:** `REJECTED AS A MAJOR OPTIMIZATION; OPT-IN MICRO-PROBE RETAINED`  
**Branch:** `exp/track-runtime`

## Question

The grouped selected-FFN path feeds cuBLAS batched matrix multiplies with
strided transpose views of the copied Qwen weights. This audit tests whether
building contiguous `[expert, input, output]` BMM operands once can remove
enough layout overhead to close the remaining sparse/dense runtime gap.

Two opt-in paths were measured:

- `grouped-prepacked`: contiguous gate, value, and output BMM operands;
- `grouped-prepacked-fused`: the same cache with the combined gate/value
  projection.

The default dispatch modes were not changed. The extra contiguous buffers are
lazy and cost device memory, so a small timing change is not sufficient for a
default switch.

## Protocol

The trained Qwen K=5 recipe is unchanged: Qwen3-0.6B, layers 19–26, `E=8`,
`K=5`, rank 1, float32, and `600/600/200` child/hard/router steps. Prefix
length is 4. Each seed uses 8 CUDA-Graph warmups and 20 timing iterations at
B1/B8/B32. Every path is compared with the same dense parent and the existing
grouped path. Eight-token fixed-shape generation is also checked.

## Results

Graph time relative to the dense parent:

| Seed | Quality CE delta | Path | B1 | B8 | B32 | Generation |
|---:|---:|:---|---:|---:|---:|:---:|
| 2026 | `+0.044972` | grouped | `1.082x` | `1.050x` | `1.069x` | exact |
| 2026 | `+0.044972` | prepacked | `1.081x` | `1.033x` | `1.065x` | exact |
| 2026 | `+0.044972` | prepacked-fused | `1.082x` | `1.030x` | `1.064x` | exact |
| 17 | `+0.039129` | grouped | `1.085x` | `1.051x` | `1.068x` | exact |
| 17 | `+0.039129` | prepacked | `1.083x` | `1.031x` | `1.065x` | exact |
| 17 | `+0.039129` | prepacked-fused | `1.083x` | `1.034x` | `1.063x` | exact |

Relative to the existing grouped path, `grouped-prepacked` is `0.984x` and
`0.982x` at B8 for seeds 2026 and 17; the B1/B32 changes are below roughly
0.5%. The fused variant is `0.982x/0.988x` at B8 versus grouped-fused. The
maximum graph/eager logit differences stay in the existing float32 range
(about `1.6e-5`), and all exact-generation checks pass.

The quality numbers are unchanged because this is an inference layout change,
not a new training or routing method. Both seeds remain inside the current
`+0.05` CE-delta quality gate, but no quality improvement is claimed.

## Decision

This is a real but small B8 micro-optimization, not the missing runtime
breakthrough. It does not justify replacing the default or paying the extra
weight-copy memory cost by default. The central gap remains the sequence of
packing, three grouped projections, and selected-output accumulation. The next
meaningful runtime step remains a genuinely tiled/fused selected-FFN kernel or
a grouped-GEMM backend that reduces those launches together.

The prepacked implementation remains opt-in as a reference for future kernel
work and as evidence that transpose-view layout alone is not the fundamental
bottleneck.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --experiment V0.234_grouped_prepacked_audit --include-grouped-prepacked --include-grouped-prepacked-fused --output results/runs/v0_234_grouped_prepacked_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --experiment V0.234_grouped_prepacked_audit --include-grouped-prepacked --include-grouped-prepacked-fused --output results/runs/v0_234_grouped_prepacked_seed17.json
```

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `tests/test_qwen_packed_dispatch.py`
- `results/runs/v0_234_grouped_prepacked_seed2026.json`
- `results/runs/v0_234_grouped_prepacked_seed17.json`
