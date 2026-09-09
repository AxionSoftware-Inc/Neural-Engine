# V0.235 — Grouped tiled selected-FFN kernel audit

**Date:** 2026-09-09  
**Status:** `REJECTED FOR RUNTIME; OPT-IN PROTOTYPE RETAINED`  
**Branch:** `exp/track-runtime`

## Question

The grouped path currently packs selected tokens and uses three cuBLAS BMMs.
This experiment tests a custom CUDA kernel that processes eight expert-major
token rows per block, reuses input/weight tiles for gate and value projection,
then reuses the SwiGLU coefficients for the output projection. It is intended
to reduce the small-BMM launch and intermediate-tensor overhead.

The kernel receives the already-packed expert-major buffer. Routing, sorting,
packing, selected-output gather, correction, and accumulation remain in the
existing path. Therefore this is a focused projection-kernel probe, not a full
fused routing implementation.

## Protocol

The trained Qwen K=5 recipe is unchanged: Qwen3-0.6B, layers 19–26, `E=8`,
`K=5`, rank 1, float32, and `600/600/200` child/hard/router steps. Prefix
length is 4. Each seed uses 8 CUDA-Graph warmups and 20 timing iterations at
B1/B8/B32. The tiled path is compared with the same dense parent and grouped
cuBLAS path. Eight-token fixed-shape generation is checked as well.

## Results

Graph time relative to the dense parent:

| Seed | Quality CE delta | Grouped | Tiled | Tiled / dense | Generation |
|---:|---:|---:|---:|---:|:---:|
| 2026 | `+0.045711` | `1.079x / 1.047x / 1.073x` | `1.841x / 1.673x / 1.434x` | B1/B8/B32 | exact |
| 17 | `+0.039675` | `1.082x / 1.050x / 1.067x` | `1.839x / 1.679x / 1.435x` | B1/B8/B32 | exact |

The tiled kernel is therefore about `1.62–1.68x` the existing grouped time at
B8 and roughly `1.43–1.51x` the dense parent at B32. It preserves the hard
route and generation tokens exactly. Full-model graph/eager logit differences
remain in the existing float32 audit range (maximum observed about `1.6e-5`).

The quality numbers are training results, not a kernel gain: the tiled path
does not change the child or router. Both seeds remain inside the current
`+0.05` CE-delta gate, but no quality improvement is claimed.

## Decision

Reject this tiled shape as a serving backend. It loses to cuBLAS because the
custom kernel has too little reuse at the tested `M` shapes and pays its own
shared-memory synchronization and tile-loop cost. Keeping it as the default
would move the project backwards.

This result is useful: the remaining gap is not solved by a simple one-block
per-expert tile over the already-packed buffer. The next kernel attempt must
either fuse packing with projection and accumulation, or use a specialized
grouped-GEMM backend with an autotuned schedule. The prior contiguous-weight
probe remains the better micro-optimization, but neither changes the default.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --experiment V0.235_grouped_tiled_kernel_audit --include-grouped-tiled --output results/runs/v0_235_grouped_tiled_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --experiment V0.235_grouped_tiled_kernel_audit --include-grouped-tiled --output results/runs/v0_235_grouped_tiled_seed17.json
```

## Artifacts

- `neural_engine/qwen_tiled_dispatch.cpp`
- `neural_engine/qwen_tiled_dispatch.cu`
- `neural_engine/qwen_tiled_dispatch.py`
- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `results/runs/v0_235_grouped_tiled_seed2026.json`
- `results/runs/v0_235_grouped_tiled_seed17.json`
