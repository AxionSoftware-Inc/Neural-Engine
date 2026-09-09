# V0.230 — Trained rank-1 shape sweep

**Date:** 2026-09-09  
**Status:** `PARITY PASS; DENSE GAP OPEN`  
**Branch:** `exp/track-runtime`

## Question

V0.229 showed that rank-1 correction survives a longer two-seed B8 audit.
This follow-up checks whether the result is specific to B8 or remains useful
across the intended fixed-shape serving points B1, B8 and B32.

## Protocol

The accepted trained Qwen K=5 recipe is unchanged: Qwen3-0.6B layers 19–26,
`E=8`, `K=5`, `subset-router`, grouped selected-FFN dispatch, float32,
correction rank 1, and child/hard/router steps `600/600/200`. Each seed uses
prefix length 4, 8 graph warmups and 20 timing iterations at B1/B8/B32.

## Results

| Seed | Quality CE delta | Top-1 agreement | B1 fused/dense | B8 fused/dense | B32 fused/dense | Generation parity |
|---:|---:|---:|---:|---:|---:|:---:|
| 2026 | `+0.047808` | `78.98%` | `1.080x` | `1.044x` | `1.065x` | exact |
| 17 | `+0.042103` | `78.71%` | `1.078x` | `1.042x` | `1.060x` | exact |

The quality gate is `CE delta <= +0.05`; both seeds pass. The sparse path is
still slower than the dense parent at every tested shape. The closest point
is B8, where the two seeds are `1.044x` and `1.042x`; B1 and B32 retain a
roughly 6–8% gap.

Grouped-fused remains effectively the same as ordinary grouped. Direct
fused/grouped graph ratios are `1.001x/1.003x/0.997x` for seed2026 and
`0.999x/0.994x/0.992x` for seed17 at B1/B8/B32. The much lower ratios versus
the single-token path only show that grouped dispatch itself is the important
optimization; grouped-fused is not a distinct additional performance win.

All four runs passed graph/eager parity and exact eight-token greedy
generation parity. Maximum final-logit error versus the single-token
reference stayed below `1.5e-5`.

## Decision

- Rank-1 is shape-robust enough to remain an opt-in runtime candidate.
- Do not make it the compatibility default; rank-64 remains the default.
- Do not add a batch-size policy for `grouped-fused`; it does not beat ordinary
  grouped consistently.
- Stop spending the next experiment on router or rank reduction. The open
  problem is selected-FFN launch/packing overhead, and the next meaningful
  patch should target a static-index/tiled or full-layer fused implementation.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --experiment V0.230_trained_rank1_shape_sweep --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --output results/runs/v0_230_rank1_shape_sweep_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --experiment V0.230_trained_rank1_shape_sweep --warmup 8 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --output results/runs/v0_230_rank1_shape_sweep_seed17.json
```

## Artifacts

- `benchmark_qwen_trained_dispatch_path_audit.py`
- `results/runs/v0_230_rank1_shape_sweep_seed2026.json`
- `results/runs/v0_230_rank1_shape_sweep_seed17.json`
- `results/RUNTIME_QWEN_RANK1_LONG_AUDIT_20260909.md`
