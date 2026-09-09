# V0.220 — Fused one-token Qwen router audit

**Date:** 2026-09-09  
**Status:** `ACCEPTED OPT-IN MICRO-OPTIMIZATION; DEFAULT UNCHANGED`  
**Branch:** `exp/track-runtime`

## Question

The stage profile showed that the standard two-linear router and hard
top-k/softmax are visible at one-token batch 1, while selected FFN dispatch
dominates at larger batch. This probe fuses the two router projections, SiLU,
top-k, and selected-weight softmax into one fixed-shape CUDA kernel. It does
not change circuit weights, correction, active K, or the default PyTorch path.

The probe uses deterministic non-tied final router weights. This is
intentional: freshly constructed test children have zero final router weights,
and PyTorch's GPU top-k tie ordering is not a semantic route that a custom
kernel should guess. A trained router must therefore be audited with its
actual non-tied weights before enabling the backend for serving.

## Results

Eight Qwen3-0.6B transferred children, float32, E=8/K=6:

| Batch | PyTorch router+top-k | Fused router+top-k | Fused/PyTorch | Fused child forward | PyTorch child forward | Child ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.1647 ms | 0.7409 ms | 0.636x | 3.3789 ms | 3.7024 ms | 0.913x |
| 8 | 1.0894 ms | 0.7349 ms | 0.675x | 19.1466 ms | 18.7007 ms | 1.024x |
| 32 | 1.2351 ms | 1.3396 ms | 1.085x | 74.0247 ms | 72.9959 ms | 1.014x |

Numerical checks passed for every batch:

- selected-id mismatches: `0` layers;
- maximum route-weight error: `1.49e-8` / `2.98e-8` / `4.47e-8`;
- maximum child-output error: `4.77e-7` / `9.54e-7` / `1.91e-6`.

The B1 route-stage reduction is real and substantial, but the end-to-end
child improvement is only about 8.7%. At B8 it regresses by about 2.4%, and
at B32 by about 1.4%. It is therefore not a general batch policy and is not a
large Qwen serving breakthrough. The independent stage profile also puts
selected FFN dispatch at about 71% of the B1 child time and about 98% at B8;
router fusion cannot solve that remaining cost.

An end-to-end one-token Qwen smoke with the same non-tied router probe was
also run after installing the children into the full model:

| Batch | PyTorch model | Fused-router model | Fused/PyTorch | Max logit error |
|---:|---:|---:|---:|---:|
| 1 | 34.2496 ms | 34.2540 ms | 1.000x | `3.81e-6` |
| 8 | 37.3064 ms | 38.0687 ms | 1.020x | `9.54e-6` |

This confirms that the router-stage win is swallowed by the rest of the
Transformer, especially attention and selected FFN work.

## Decision

- Keep `single_token_router_backend="torch"` as the default.
- Keep `"cuda-fused"` as an opt-in fixed-shape one-token probe.
- Do not enable it for tied/zero routers without a defined tie-compatible
  route contract.
- Do not claim that router fusion makes the whole Transformer faster; Qwen
  attention remains outside this child-level measurement and selected FFN
  dispatch remains the larger cost at B8.
- The next runtime work should target a trained-child CUDA Graph/e2e audit or
  stop pursuing router-only fusion and focus on selected FFN dispatch.

## Reproduction

```text
python -u benchmark_qwen_fused_router_audit.py --warmup 10 --iterations 30 --batch-sizes 1 8 32 --output results/runs/v0_220_fused_router_audit.json
```

The full-model B1/B8 smoke is stored separately:

```text
python -u benchmark_qwen_fused_router_audit.py --warmup 8 --iterations 20 --batch-sizes 1 8 --output results/runs/v0_220_fused_router_audit_e2e.json
```

The earlier stage decomposition is in
`results/runs/v0_219_router_topk_dispatch_profile.json` and is reproduced by:

```text
python -u benchmark_qwen_router_topk_profile.py --warmup 15 --iterations 60 --batch-sizes 1 8 32 --output results/runs/v0_219_router_topk_dispatch_profile.json
```

## Artifacts

- `neural_engine/qwen_router_dispatch.py`
- `neural_engine/qwen_router_dispatch.cpp`
- `neural_engine/qwen_router_dispatch.cu`
- `benchmark_qwen_fused_router_audit.py`
- `benchmark_qwen_router_topk_profile.py`
- integration hook: `benchmark_qwen_multi_layer_transplant.py`
