# V0.245 — Atomic pack plus folded-output combination audit

**Date:** 2026-09-09  
**Status:** `PARITY PASS; ADDITIVE SMALL SPEEDUP; DEFAULT UNCHANGED`  
**Branch:** `exp/track-runtime`

## Purpose

V0.243/V0.244 established a small, shape-robust gain from CUDA atomic route
packing. V0.242 established a separate sub-1% gain from folding the exact
low-rank correction into the per-expert output projection. This experiment
checks whether the two independent runtime changes compose cleanly.

The probe uses `grouped-adaptive-atomic-effective-output`: CUDA atomic packing
plus the cached effective output weight

```text
W_effective[e] = W_out[e] + mix_out[e] @ mix_in[e] @ W_out[e]
```

Routing, active K, expert weights, correction values, accumulation scale, and
quality training are unchanged.

## Protocol

Local `Qwen/Qwen3-0.6B`, layers `19–26`, `E=8`, `K=5`, rank-1 correction,
float32 CUDA, and the `600/600/200` trained-cascade recipe. Prefix length was
4; batch sizes were 1, 8, and 32. Each path used 10 CUDA-Graph warmups and 20
timing iterations. The baseline was `grouped-adaptive`; comparison paths were
`grouped-adaptive-atomic-pack` and the combined probe. Two seeds used the same
protocol.

Graph ratio is measured against `grouped-adaptive`; a negative change means
the probe is faster.

| seed | batch | adaptive | atomic pack | atomic + folded output | combo/adaptive | change |
|---:|---:|---:|---:|---:|---:|---:|
| 2026 | 1 | 11.222 | 10.898 | 10.839 | 0.9658 | −3.42% |
| 2026 | 8 | 17.064 | 16.627 | 16.536 | 0.9691 | −3.09% |
| 2026 | 32 | 34.666 | 34.061 | 33.931 | 0.9788 | −2.12% |
| 17 | 1 | 11.195 | 11.002 | 10.833 | 0.9677 | −3.23% |
| 17 | 8 | 17.061 | 16.652 | 16.609 | 0.9735 | −2.65% |
| 17 | 32 | 34.561 | 34.029 | 33.907 | 0.9811 | −1.89% |

The exact ratios above are the decision metric; the source JSON files contain
the raw millisecond values. Across both seeds, the combined path is faster at
all six points. Relative to atomic pack alone, the folded-output addition is
small and variable (`0.26–1.54%`), so the changes are complementary but not
strongly additive.

## Correctness and quality

Both seeds returned `PARITY_PASS`. CUDA-Graph/eager checks passed and eight-
token greedy generation matched the grouped baseline exactly. The maximum
logit error stayed within the existing float32 grouped-dispatch tolerance.
Quality remains the same trained-cascade result; this is a runtime-only probe
and does not claim a CE or routing improvement.

## Decision

- Keep the combined path as an opt-in runtime configuration for controlled
  experiments.
- Keep the default unchanged: the improvement is only about 1.9–3.4% and the
  grouped implementation is still near or above dense latency at these shapes.
- Treat the two components as reusable baselines for the next material kernel
  experiment, not as the final serving architecture.
- The next material target remains a route-aware fused kernel that avoids the
  intermediate expert-major workspace and combines route, projection,
  correction, and token accumulation more directly.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --warmup 10 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-atomic-pack --include-grouped-adaptive-atomic-effective-output --experiment V0.245_atomic_effective_combo_probe --output results/runs/v0_245_atomic_effective_combo_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --warmup 10 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-atomic-pack --include-grouped-adaptive-atomic-effective-output --experiment V0.245_atomic_effective_combo_probe --output results/runs/v0_245_atomic_effective_combo_seed17.json
```

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `results/runs/v0_245_atomic_effective_combo_seed2026.json`
- `results/runs/v0_245_atomic_effective_combo_seed17.json`
