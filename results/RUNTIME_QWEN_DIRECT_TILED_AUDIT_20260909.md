# V0.248 — Route-aware direct tiled dispatch audit

**Date:** 2026-09-09  
**Status:** `SPEED HYPOTHESIS REJECTED; PARITY-SAFE OPT-IN RETAINED`  
**Branch:** `exp/track-runtime`

## Question

V0.240 identified expert-major packing as the largest grouped-path stage.
V0.247 removed only the final gather/accumulation overhead and produced a
small stable speed gain. This probe tests the stronger hypothesis that route
packing, grouped SwiGLU projection, and token accumulation should be fused in
one token-tile CUDA kernel.

The new kernel uses sorted route metadata to read token rows directly from
the flat hidden buffer. One block owns an expert and eight expert-local rows;
it computes tiled gate/value/output projections and atomically accumulates
the weighted result into token-major output. It therefore avoids materializing
the grouped hidden and grouped output workspaces.

## Implementation correction

The first Qwen attempt exposed an invalid route read on partially filled final
tiles. The row-validity check was corrected to use the tile-local row index.
The partial-tile CUDA parity test then passed with maximum absolute error
`1.83e-4`; this is within the probe's float32 tolerance. The real benchmark
was rerun from a clean extension cache after that fix.

## Protocol

The benchmark uses local `Qwen/Qwen3-0.6B`, layers `19--26`, `E=8`, `K=5`,
rank-1 correction, float32 CUDA, and the trained-cascade recipe
`600/600/200`. Prefix length is 4; batch sizes are 1, 8, and 32. Each path
uses 5 CUDA-Graph warmups and 10 timing iterations. Seeds 2026 and 17 use
the same protocol.

The baseline is `grouped-adaptive`; the new path is
`grouped-adaptive-direct-tiled`. The comparison also ran the V0.245 and
V0.247 grouped atomic paths as controls.

## CUDA-Graph latency

Values are milliseconds; lower is better. The change is relative to the
`grouped-adaptive` baseline.

| seed | batch | grouped-adaptive | direct-tiled | change |
|---:|---:|---:|---:|---:|
| 2026 | 1 | 11.193 | 18.721 | +67.3% |
| 2026 | 8 | 16.936 | 27.757 | +63.9% |
| 2026 | 32 | 34.528 | 44.479 | +28.8% |
| 17 | 1 | 11.195 | 18.725 | +67.3% |
| 17 | 8 | 16.980 | 27.481 | +61.8% |
| 17 | 32 | 34.524 | 43.904 | +27.2% |
| **two-seed mean** | **1** | **11.194** | **18.723** | **+67.3%** |
| **two-seed mean** | **8** | **16.958** | **27.619** | **+62.9%** |
| **two-seed mean** | **32** | **34.526** | **44.192** | **+28.0%** |

The result is consistently negative. Eliminating the explicit pack workspace
does not eliminate the work: the route-aware custom kernel loses the efficient
batched GEMM behavior of the grouped path and adds atomic token accumulation.
The loss is largest at B1/B8, where the tile has little reuse, and remains
material at B32.

## Correctness and quality

- Both seeds returned `PARITY_PASS`.
- CUDA-Graph/eager checks passed within the existing float32 tolerance.
- Eight-token greedy generation matched the grouped baseline exactly for both
  seeds.
- The independent partial-tile smoke test passed with max absolute error
  `1.83e-4`.
- Quality was unchanged by the dispatch experiment. CE deltas versus the
  dense reference were `+0.04863` for seed 2026 and `+0.04049` for seed 17.

## Decision

- Reject `grouped-adaptive-direct-tiled` as a serving-speed solution and do
  not make it the default.
- Retain it as an opt-in parity reference for future kernel work; its fused
  route/output contract is useful, but this schedule is not competitive.
- Do not pursue more variants that replace grouped cuBLAS GEMM with the same
  one-block-per-expert-row custom schedule.
- The next meaningful runtime attempt needs a true grouped-GEMM backend or a
  token-tile kernel that preserves high-throughput matrix-multiply reuse,
  rather than only moving pack/finalize operations into one launch.

This is a runtime result only. It does not improve routing, capacity scaling,
or the quality of the attention-free architecture.

## Reproduction

```text
$env:TORCH_EXTENSIONS_DIR = 'C:\\Users\\shaxz\\AppData\\Local\\Temp\\neural_engine_torch_ext_v248_fix'
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --warmup 5 --iterations 10 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-atomic-effective-output --include-grouped-adaptive-atomic-finalize --include-grouped-adaptive-direct-tiled --experiment V0.248_direct_tiled_probe --output results/runs/v0_248_direct_tiled_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --warmup 5 --iterations 10 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-atomic-effective-output --include-grouped-adaptive-atomic-finalize --include-grouped-adaptive-direct-tiled --experiment V0.248_direct_tiled_probe --output results/runs/v0_248_direct_tiled_seed17.json
```

## Artifacts

- `neural_engine/qwen_direct_tiled_dispatch.py`
- `neural_engine/qwen_direct_tiled_dispatch.cpp`
- `neural_engine/qwen_direct_tiled_dispatch.cu`
- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `results/runs/v0_248_direct_tiled_seed2026.json`
- `results/runs/v0_248_direct_tiled_seed17.json`
