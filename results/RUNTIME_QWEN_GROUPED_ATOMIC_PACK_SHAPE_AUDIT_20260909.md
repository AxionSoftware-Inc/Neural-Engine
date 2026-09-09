# V0.244 — CUDA atomic route-pack shape sweep

**Date:** 2026-09-09  
**Status:** `PARITY PASS; SHAPE-ROBUST SMALL SPEEDUP; DEFAULT UNCHANGED`  
**Branch:** `exp/track-runtime`

## Purpose

V0.243 showed a repeatable 1.4–2.6% speedup from replacing the grouped
dispatch `argsort`/workspace pack with a CUDA atomic pack kernel. This follow-up
checks whether the signal is limited to the original prefix length of 4 or
survives across the fixed-shape paths used by the runtime.

## Protocol

Local `Qwen/Qwen3-0.6B`, layers `19–26`, `E=8`, `K=5`, rank-1 correction,
float32 CUDA, the `600/600/200` trained-cascade recipe, seed `2026`, batch
sizes `1/8/32`, and prefix lengths `1/4/8`. Each path used 15 CUDA-Graph
warmups and 30 timing iterations. Baseline: `grouped-adaptive`. Probe:
`grouped-adaptive-atomic-pack`.

Graph ratio is `atomic-pack / adaptive`; a negative change means the atomic
pack path is faster.

| prefix | batch | adaptive/dense | atomic/dense | atomic/adaptive | change |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1.07501 | 1.04480 | 0.97190 | −2.81% |
| 1 | 8 | 1.01598 | 1.00174 | 0.98598 | −1.40% |
| 1 | 32 | 1.05265 | 1.04124 | 0.98916 | −1.08% |
| 4 | 1 | 1.07619 | 1.04501 | 0.97103 | −2.90% |
| 4 | 8 | 1.01867 | 1.00048 | 0.98214 | −1.79% |
| 4 | 32 | 1.05606 | 1.03938 | 0.98421 | −1.58% |
| 8 | 1 | 1.07046 | 1.05069 | 0.98154 | −1.85% |
| 8 | 8 | 1.00911 | 0.99171 | 0.98276 | −1.72% |
| 8 | 32 | 1.05196 | 1.04162 | 0.99017 | −0.98% |

The probe is faster at all nine points, with an observed improvement range of
`0.98–2.90%`. It does not close the dense gap: the atomic grouped path is
still approximately equal to or slower than dense at these shapes.

## Correctness

The run returned `PARITY_PASS`. CUDA-Graph/eager parity checks passed, the
maximum logit error stayed in the existing float32 tolerance range, and
eight-token greedy generation matched the grouped baseline exactly.

## Decision

- The V0.243 signal is not a prefix-4 artifact; retain the atomic pack path as
  a validated opt-in runtime baseline.
- Keep the default unchanged because the gain is small and the intermediate
  expert-major workspace remains.
- Proceed to the next material probe: a route-aware fused kernel that writes
  final token outputs directly, bypassing the packed workspace and the current
  separate select/correction/accumulate stages.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --warmup 15 --iterations 30 --batch-sizes 1 8 32 --prefix-lengths 1 4 8 --include-grouped-adaptive --include-grouped-adaptive-atomic-pack --experiment V0.244_grouped_atomic_pack_shape_sweep --output results/runs/v0_244_grouped_atomic_pack_shape_seed2026.json
```

## Artifact

- `results/runs/v0_244_grouped_atomic_pack_shape_seed2026.json`
