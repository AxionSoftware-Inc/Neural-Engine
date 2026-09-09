# V0.219 — Qwen router/top-k/dispatch stage profile

**Date:** 2026-09-09  
**Status:** `DIAGNOSTIC BASELINE`  
**Branch:** `exp/track-runtime`

## Result

The fixed hidden-state profile measures eight transferred Qwen3-0.6B
children, E=8/K=6, float32. It excludes Transformer attention and KV-cache
work so the child stages can be compared directly.

| Batch | Router MLP | Top-k + softmax | Router + top-k | Selected FFN dispatch | Full child |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.6255 ms | 0.4305 ms | 1.3911 ms | 2.5532 ms | 3.5784 ms |
| 8 | 0.5413 ms | 0.4036 ms | 1.0079 ms | 17.9253 ms | 18.2360 ms |
| 32 | 0.7242 ms | 0.2931 ms | 1.1795 ms | 70.9509 ms | 71.5466 ms |

The decomposed sum is slightly above the full forward (`1.102x/1.038x/1.008x`)
because the separate probes have different allocator/cache behavior; it is a
stage attribution, not an additive end-to-end timing claim.

## Interpretation

At B1 the router and selection logic are material, but selected FFN dispatch
still dominates the child. At B8 and B32 dispatch is effectively the whole
child cost. This rules out a router-only explanation for the one-token
selected-child slowdown at larger batch. It motivates V0.220's fused-router
probe but also limits the expected end-to-end gain.

## Reproduction

```text
python -u benchmark_qwen_router_topk_profile.py --warmup 15 --iterations 60 --batch-sizes 1 8 32 --output results/runs/v0_219_router_topk_dispatch_profile.json
```
