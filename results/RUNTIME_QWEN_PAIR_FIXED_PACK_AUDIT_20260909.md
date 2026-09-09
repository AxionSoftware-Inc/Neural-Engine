# V0.250 — Pair-based deterministic fixed-layout pack audit

**Date:** 2026-09-09  
**Status:** `PARITY PASS; SMALL STABLE RUNTIME GAIN; OPT-IN ONLY`  
**Branch:** `exp/track-runtime`

## Question

V0.249 replaced the grouped route-pack sort/index path with a deterministic
fixed `[expert, token, hidden]` layout, but its kernel launched one block for
every expert-token row and scanned the K selected experts. This probe removes
that unnecessary work.

V0.250 launches one CUDA block per selected `(token, slot)` pair and writes
directly to `expert * token_count + token`. Top-k expert IDs are unique per
token, so no atomic counter, scan, or sort is needed. Unselected output rows
are intentionally left unspecified: the grouped consumer reads only the
selected `grouped_indices` rows. Grouped projections, folded correction,
CUDA finalization, router, model weights, and the public API are unchanged.

This report supersedes V0.249's implementation-level description. The V0.249
report remains the historical result for the previous scan-and-zero kernel.

## Protocol

The benchmark uses local `Qwen/Qwen3-0.6B`, layers `19--26`, `E=8`, `K=5`,
rank-1 correction, float32 CUDA, and the trained-cascade recipe
`600/600/200`. Prefix length is 4; batch sizes are 1, 8, and 32. Each path
uses 20 CUDA-Graph warmups and 50 timing iterations. Seeds 2026 and 17 use
the same protocol.

Baseline: `grouped-adaptive`. Control: V0.247
`grouped-adaptive-atomic-finalize`. Probe:
`grouped-adaptive-fixed-pack` (pair-based fixed pack plus folded output and
CUDA finalization).

## CUDA-Graph latency

Values are milliseconds; lower is better. The change is relative to the
`grouped-adaptive` baseline.

| seed | batch | grouped-adaptive | atomic-finalize | pair fixed-pack | fixed change |
|---:|---:|---:|---:|---:|---:|
| 2026 | 1 | 11.162 | 10.744 | 10.728 | -3.89% |
| 2026 | 8 | 16.917 | 16.545 | 16.555 | -2.14% |
| 2026 | 32 | 34.524 | 33.840 | 33.817 | -2.05% |
| 17 | 1 | 11.186 | 10.753 | 10.717 | -4.19% |
| 17 | 8 | 16.978 | 16.544 | 16.565 | -2.43% |
| 17 | 32 | 34.570 | 33.949 | 33.962 | -1.76% |
| **two-seed mean** | **1** | **11.174** | **10.749** | **10.723** | **-4.04%** |
| **two-seed mean** | **8** | **16.948** | **16.545** | **16.560** | **-2.29%** |
| **two-seed mean** | **32** | **34.547** | **33.895** | **33.890** | **-1.90%** |

Against atomic-finalize, the pair fixed-pack two-seed mean changes by
`-0.24%` at B1, `+0.09%` at B8, and `-0.01%` at B32. Thus the pair mapping
removes the atomic counter without creating a new runtime tier; its practical
speed is equivalent to the existing atomic-finalize control.

## Correctness and quality

- Both long runs returned `PARITY_PASS`.
- CUDA-Graph/eager checks stayed within the existing float32 tolerance.
- Eight-token greedy generation matched the grouped adaptive fixed-pack path
  exactly for both seeds.
- A standalone 13-token random top-k smoke test found exact equality on every
  selected destination (`max_abs=0`). The smoke test correctly does not assert
  zeroes in unselected destinations because V0.250 no longer initializes them.
- Quality was unchanged by the dispatch experiment. CE deltas versus the
  dense reference were `+0.044987` for seed 2026 and `+0.040411` for seed 17
  (mean `+0.042699`).

Raw long-run records:

- `runs/v0_250_pair_fixed_pack_long_seed2026.json`
- `runs/v0_250_pair_fixed_pack_long_seed17.json`

## Decision

- Keep `grouped-adaptive-fixed-pack` as an opt-in runtime candidate.
- Do not make it the default: the gain over adaptive packing is modest, and
  it is effectively tied with V0.247 atomic-finalize.
- The result validates the fixed-layout pair mapping and rules out further
  pack-only micro-variants as the likely source of a large serving gain.
- Next runtime work should target grouped matrix-multiply reuse/autotuning or
  another backend-level optimization. More pack kernel variations are low
  priority unless profiling identifies a new bottleneck.

## Reproduction

```powershell
$env:TORCH_EXTENSIONS_DIR='C:\Users\shaxz\AppData\Local\Temp\neural_engine_torch_ext_v250'
python -u benchmark_qwen_trained_dispatch_path_audit.py `
  --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 `
  --seed 17 --warmup 20 --iterations 50 --batch-sizes 1 8 32 `
  --prefix-lengths 4 --include-grouped-adaptive `
  --include-grouped-adaptive-atomic-finalize `
  --include-grouped-adaptive-fixed-pack `
  --experiment V0.250_pair_fixed_pack_long `
  --output results/runs/v0_250_pair_fixed_pack_long_seed17.json
```
