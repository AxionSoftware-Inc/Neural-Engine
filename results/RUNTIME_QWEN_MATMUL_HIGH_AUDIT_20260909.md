# V0.251 — Float32 matmul precision A/B audit

**Date:** 2026-09-09  
**Status:** `PERFORMANCE SIGNAL; NUMERICAL GATE FAIL; OPT-IN ONLY`  
**Branch:** `exp/track-runtime`

## Question

The grouped path is dominated by float32 batched matrix multiplies. This probe
tests PyTorch's CUDA float32 matmul precision setting as a serving-only A/B:
the child/router training recipe remains at `highest`, then evaluation and
latency measurement use `high`. No route, expert weight, model architecture,
or active parameter count changes.

The benchmark now records `matmul_precision` and computes its status from a
real graph/eager numerical gate instead of unconditionally writing
`PARITY_PASS`.

## Protocol

Local `Qwen/Qwen3-0.6B`, layers `19--26`, `E=8`, `K=5`, rank-1 correction,
float32 CUDA, and trained-cascade recipe `600/600/200`. Prefix length is 4;
batch sizes are 1, 8, and 32. Each path uses 20 CUDA-Graph warmups and 50
timing iterations. Seeds 2026 and 17 use the same protocol.

Baseline/control paths are `grouped-adaptive`,
`grouped-adaptive-atomic-finalize`, and `grouped-adaptive-fixed-pack`.
V0.250's `highest` measurements are the strict-precision comparison.

## CUDA-Graph latency at `high`

Values are milliseconds; lower is better.

| seed | path | B1 | B8 | B32 |
|---:|---|---:|---:|---:|
| 2026 | dense | 10.375 | 15.142 | 26.614 |
| 2026 | grouped-adaptive | 11.163 | 15.814 | 27.625 |
| 2026 | atomic-finalize | 10.755 | 15.378 | 26.895 |
| 2026 | fixed-pack | 10.713 | 15.356 | 26.921 |
| 17 | dense | 10.410 | 15.161 | 26.678 |
| 17 | grouped-adaptive | 11.180 | 15.775 | 27.603 |
| 17 | atomic-finalize | 10.773 | 15.347 | 26.914 |
| 17 | fixed-pack | 10.724 | 15.357 | 26.905 |

Relative to V0.250's `highest` fixed-pack path, the `high` fixed-pack
two-seed mean changes by approximately `0.0%` at B1, `-7.3%` at B8, and
`-20.6%` at B32. Relative to the dense `high` path, fixed-pack is only about
`+3.1% / +1.4% / +1.0%` at B1/B8/B32. This is a real throughput signal, not
a quality or routing improvement.

## Numerical and quality checks

- Seed 2026: `PARITY_FAIL`; maximum graph/eager logit error `0.007687`,
  versus the `0.001` gate.
- Seed 17: `PARITY_FAIL`; maximum graph/eager logit error `0.007801`,
  versus the `0.001` gate.
- Eight-token greedy generation stayed exactly equal for the tested paths in
  both seeds, but that short check does not override the failed numerical
  gate.
- Quality CE deltas versus the dense reference were `+0.047031` and
  `+0.042997` (mean `+0.045014`). They are not better than V0.250's
  `+0.044987` and `+0.040411`; no quality gain is claimed.

Raw records:

- `runs/v0_251_matmul_high_seed2026.json`
- `runs/v0_251_matmul_high_seed17.json`

## Decision

- Do not enable `matmul_precision=high` by default. The numerical gate fails
  reproducibly on both seeds despite the attractive B8/B32 speedup.
- Retain the CLI option as an explicit opt-in for deployments that accept
  TF32-like float32 drift and validate their own output tolerance.
- Strict reproducibility and the quality benchmark remain on `highest`.
- The next runtime experiment should return to grouped-GEMM scheduling/reuse;
  another precision knob is unlikely to solve the architectural bottleneck.

## Reproduction

```powershell
$env:TORCH_EXTENSIONS_DIR='C:\Users\shaxz\AppData\Local\Temp\neural_engine_torch_ext_v250'
python -u benchmark_qwen_trained_dispatch_path_audit.py `
  --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 `
  --seed 2026 --matmul-precision high --warmup 20 --iterations 50 `
  --batch-sizes 1 8 32 --prefix-lengths 4 `
  --include-grouped-adaptive --include-grouped-adaptive-atomic-finalize `
  --include-grouped-adaptive-fixed-pack `
  --experiment V0.251_matmul_high_probe `
  --output results/runs/v0_251_matmul_high_seed2026.json
```
