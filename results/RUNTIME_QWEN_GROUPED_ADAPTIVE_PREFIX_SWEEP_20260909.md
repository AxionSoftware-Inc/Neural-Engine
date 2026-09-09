# V0.239 — Adaptive grouped prefix/shape sweep

**Date:** 2026-09-09  
**Status:** `PREFIX-SHAPE PASS; OPT-IN ONLY`  
**Branch:** `exp/track-runtime`

## Purpose

V0.238 showed that selecting the plain grouped backend for one-row decode and
the combined cached/prepacked/fused backend for larger row counts is currently
the best runtime policy. This sweep checks whether that rule survives changes
to the KV-cache prefix length rather than only the four-token smoke.

The model and route are unchanged. The policy is still based only on the
current flattened FFN row count, not on task difficulty or an imposed active
parameter count.

## Protocol

Qwen3-0.6B, layers 19–26, `E=8`, `K=5`, rank 1, float32, and
`600/600/200` child/hard/router steps. Prefix lengths are 4, 32, and 128;
decode batch sizes are 1 and 8. Each seed uses 8 CUDA-Graph warmups and 20
timing iterations per shape. Dense parent, plain grouped, and adaptive paths
are measured in the same process. Eight-token fixed-shape generation and
full-model graph/eager parity are checked.

## Results

Adaptive graph time relative to the dense parent, shown as
`prefix 4 / prefix 32 / prefix 128`:

| Seed | Quality CE delta | B1 | B8 | Generation |
|---:|---:|:---|:---|:---:|
| 2026 | `+0.045993` | `1.052x / 1.054x / 1.060x` | `1.001x / 1.016x / 0.988x` | exact |
| 17 | `+0.040353` | `1.079x / 1.073x / 1.065x` | `1.013x / 1.013x / 1.007x` | exact |

The adaptive path remains close to dense at B8 across all tested prefixes and
is better than the plain grouped path at every B8/prefix point in both seeds.
At B1 it removes the extra optimized-path overhead, but the remaining gap is
still about 5–8% relative to dense. Maximum graph/eager logit differences stay
within the existing float32 audit range (about `1.3e-5`), and generation
tokens match exactly.

The quality deltas remain within the current `+0.05` gate. This sweep changes
runtime backend selection only and provides no quality gain.

## Decision

The row-count adaptive rule is robust enough to retain as the current best
opt-in decode policy. It is not promoted to a global default because the
small-batch gap is still open and prefill lengths beyond 128, other batch
sizes, and a second GPU have not been audited.

The next meaningful runtime work remains a route-count-aware packing path or
autotuned grouped GEMM. The adaptive policy should be used as the selection
layer for that backend rather than adding more independent Python shortcuts.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 8 --iterations 20 --batch-sizes 1 8 --prefix-lengths 4 32 128 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --experiment V0.239_grouped_adaptive_prefix_sweep --include-grouped-adaptive --output results/runs/v0_239_grouped_adaptive_prefix_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 8 --iterations 20 --batch-sizes 1 8 --prefix-lengths 4 32 128 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --experiment V0.239_grouped_adaptive_prefix_sweep --include-grouped-adaptive --output results/runs/v0_239_grouped_adaptive_prefix_seed17.json
```

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `tests/test_qwen_packed_dispatch.py`
- `results/runs/v0_239_grouped_adaptive_prefix_seed2026.json`
- `results/runs/v0_239_grouped_adaptive_prefix_seed17.json`
