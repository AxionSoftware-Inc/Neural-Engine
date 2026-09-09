# V0.249 — Deterministic fixed-layout route-pack audit

**Date:** 2026-09-09  
**Status:** `PARITY PASS; SMALL STABLE RUNTIME GAIN; OPT-IN ONLY`  
**Branch:** `exp/track-runtime`

## Question

The V0.240 profile showed that grouped route packing was the largest
instrumented stage. V0.243's CUDA atomic pack removed the PyTorch sort/index
copy sequence and gave a small gain, but it still used an expert-local atomic
counter. This probe tests a deterministic alternative.

The new packer writes to a fixed `[expert, token, hidden]` layout. Each
`[expert, token]` block scans the token's K selected experts and copies the
hidden row if that expert is present, otherwise it writes zero. The pair
mapping is then the deterministic formula `expert * token_count + token`.
The existing grouped cuBLAS projections, folded correction, V0.247 CUDA
finalizer, router, and model weights are unchanged.

## Protocol

The benchmark uses local `Qwen/Qwen3-0.6B`, layers `19--26`, `E=8`, `K=5`,
rank-1 correction, float32 CUDA, and the trained-cascade recipe
`600/600/200`. Prefix length is 4; batch sizes are 1, 8, and 32. Each path
uses 20 CUDA-Graph warmups and 50 timing iterations. Seeds 2026 and 17 use
the same protocol.

Baseline: `grouped-adaptive`. Control: V0.247
`grouped-adaptive-atomic-finalize`. Probe:
`grouped-adaptive-fixed-pack` (fixed pack plus folded output and CUDA
finalization).

## CUDA-Graph latency

Values are milliseconds; lower is better. The change is relative to the
`grouped-adaptive` baseline.

| seed | batch | grouped-adaptive | atomic-finalize | fixed-pack | fixed change |
|---:|---:|---:|---:|---:|---:|
| 2026 | 1 | 11.182 | 10.760 | 10.719 | -4.15% |
| 2026 | 8 | 16.987 | 16.544 | 16.547 | -2.59% |
| 2026 | 32 | 34.612 | 33.865 | 34.002 | -1.76% |
| 17 | 1 | 11.198 | 10.759 | 10.711 | -4.34% |
| 17 | 8 | 16.969 | 16.556 | 16.540 | -2.53% |
| 17 | 32 | 34.518 | 33.815 | 33.874 | -1.87% |
| **two-seed mean** | **1** | **11.190** | **10.760** | **10.715** | **-4.24%** |
| **two-seed mean** | **8** | **16.978** | **16.550** | **16.543** | **-2.56%** |
| **two-seed mean** | **32** | **34.565** | **33.840** | **33.938** | **-1.81%** |

Relative to V0.247 atomic-finalize, fixed-pack changes the two-seed mean by
`-0.42%` at B1, `-0.04%` at B8, and `+0.29%` at B32. It is therefore a
deterministic alternative with essentially the same performance, not a new
large speed tier. The sparse path remains near the dense path at these
shapes; this does not solve the overall serving-cost gap.

## Correctness and quality

- Both long runs returned `PARITY_PASS`.
- CUDA-Graph/eager checks passed within the existing float32 tolerance.
- Eight-token greedy generation matched the grouped baseline exactly for both
  seeds.
- An independent 13-token pack smoke test produced exact equality (`max_abs=0`)
  against a reference fixed-layout implementation.
- Quality was unchanged by the dispatch experiment. CE deltas versus the
  dense reference were `+0.04580` for seed 2026 and `+0.03681` for seed 17.

## Decision

- Retain `grouped-adaptive-fixed-pack` as an opt-in runtime candidate. It is
  deterministic and removes the atomic counter while preserving grouped GEMM
  reuse.
- Do not make it the default: the gain is small and it is not consistently
  better than V0.247 at every batch shape.
- V0.248 direct-tiled dispatch remains rejected; replacing grouped cuBLAS
  projection with the current custom one-block schedule was substantially
  slower.
- Further pack-only variants are low priority. The next material runtime
  experiment must improve grouped matrix-multiply reuse or use an autotuned
  grouped-GEMM backend.

This is a runtime result only. It does not improve routing, capacity scaling,
or the quality of the attention-free architecture.

## Reproduction

```text
$env:TORCH_EXTENSIONS_DIR = 'C:\\Users\\shaxz\\AppData\\Local\\Temp\\neural_engine_torch_ext_v249'
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --warmup 20 --iterations 50 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-atomic-finalize --include-grouped-adaptive-fixed-pack --experiment V0.249_deterministic_pack_long --output results/runs/v0_249_deterministic_pack_long_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --warmup 20 --iterations 50 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-atomic-finalize --include-grouped-adaptive-fixed-pack --experiment V0.249_deterministic_pack_long --output results/runs/v0_249_deterministic_pack_long_seed17.json
```

## Artifacts

- `neural_engine/qwen_deterministic_pack.py`
- `neural_engine/qwen_deterministic_pack.cpp`
- `neural_engine/qwen_deterministic_pack.cu`
- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `results/runs/v0_249_deterministic_pack_long_seed2026.json`
- `results/runs/v0_249_deterministic_pack_long_seed17.json`
