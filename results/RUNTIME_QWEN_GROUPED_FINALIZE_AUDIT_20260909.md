# V0.247 — Grouped folded-output CUDA finalization audit

**Date:** 2026-09-09  
**Status:** `PARITY PASS; SMALL STABLE RUNTIME GAIN; QUALITY DEFAULT UNCHANGED`  
**Branch:** `exp/track-runtime`

## Question

V0.245 showed that atomic route packing plus folded output correction was a
small but repeatable improvement. V0.246 then tested a direct one-block-per-
route fused kernel and rejected it: removing the grouped workspace lost the
batched GEMM reuse and made the path 34--140% slower.

This probe keeps the useful grouped projection and folded correction from
V0.245, but replaces the final Python/PyTorch gather and `index_add` with one
CUDA kernel. The kernel writes directly to the token output and therefore
tests whether the remaining grouped-finalization overhead is material.

The probe does not change the router, route IDs, active K, expert weights,
training recipe, or quality path. It is inference-only and opt-in.

## Protocol

The benchmark uses local `Qwen/Qwen3-0.6B`, layers `19--26`, `E=8`, `K=5`,
rank-1 correction, float32 CUDA, and the trained-cascade recipe
`600/600/200`. Prefix length is 4; batch sizes are 1, 8, and 32. Each path
uses 20 CUDA-Graph warmups and 50 timing iterations. Seeds 2026 and 17 use
the same protocol.

The baseline is `grouped-adaptive`. `atomic-effective-output` is the V0.245
combined path. `atomic-finalize` is the new path.

## CUDA-Graph latency

Values are milliseconds; lower is better. Ratios and changes are relative to
the `grouped-adaptive` baseline.

| seed | batch | grouped-adaptive | atomic-effective-output | atomic-finalize | finalize change |
|---:|---:|---:|---:|---:|---:|
| 2026 | 1 | 11.186 | 10.797 | 10.792 | -3.52% |
| 2026 | 8 | 17.024 | 16.550 | 16.581 | -2.61% |
| 2026 | 32 | 34.692 | 33.966 | 33.898 | -2.29% |
| 17 | 1 | 11.196 | 10.826 | 10.739 | -4.08% |
| 17 | 8 | 17.054 | 16.596 | 16.542 | -3.00% |
| 17 | 32 | 34.581 | 34.084 | 33.934 | -1.87% |
| **two-seed mean** | **1** | **11.191** | **10.811** | **10.765** | **-3.80%** |
| **two-seed mean** | **8** | **17.039** | **16.573** | **16.561** | **-2.80%** |
| **two-seed mean** | **32** | **34.637** | **34.025** | **33.916** | **-2.08%** |

The new finalizer improves the V0.245 combined path by only about `0.07%`
at batch 8 on the two-seed mean, and by `0.32--0.43%` at batches 1 and 32.
Thus the gain is real enough to retain as a runtime optimization, but it is
not a large architectural or scaling result. The grouped projection remains
the dominant reusable computation.

## Correctness and quality

- Both long runs returned `PARITY_PASS`.
- CUDA-Graph/eager checks passed within the existing float32 tolerance.
- Eight-token greedy generation matched the grouped baseline exactly for both
  seeds.
- An isolated random CUDA smoke test matched PyTorch `index_add` with maximum
  absolute error `2.38e-7`.
- The quality recipe was unchanged. CE deltas versus the dense reference were
  `+0.04487` for seed 2026 and `+0.04190` for seed 17. The finalizer changes
  dispatch mechanics only; it does not claim a quality improvement.

## Decision

- Retain `grouped-adaptive-atomic-finalize` as an opt-in runtime path.
- Do not make it the quality or general model default: it does not alter the
  learned circuit selection and gives no quality gain.
- Keep V0.246's direct fused folded-output kernel rejected as a speed path.
- Do not spend more time on gather/index-add removal alone. A material next
  speed step must attack grouped projection with tiled/token-tile GEMM reuse,
  not merely another output materialization variant.
- The next benchmark should compare that kernel at the same fixed protocol;
  model-capacity experiments remain separate from runtime work.

## Reproduction

```text
$env:TORCH_EXTENSIONS_DIR = 'C:\\Users\\shaxz\\AppData\\Local\\Temp\\neural_engine_torch_ext_v247'
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --warmup 20 --iterations 50 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-atomic-effective-output --include-grouped-adaptive-atomic-finalize --experiment V0.247_grouped_finalize_long --output results/runs/v0_247_grouped_finalize_long_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --warmup 20 --iterations 50 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-atomic-effective-output --include-grouped-adaptive-atomic-finalize --experiment V0.247_grouped_finalize_long --output results/runs/v0_247_grouped_finalize_long_seed17.json
```

## Artifacts

- `neural_engine/qwen_grouped_finalize.py`
- `neural_engine/qwen_grouped_finalize.cpp`
- `neural_engine/qwen_grouped_finalize.cu`
- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `results/runs/v0_247_grouped_finalize_long_seed2026.json`
- `results/runs/v0_247_grouped_finalize_long_seed17.json`
