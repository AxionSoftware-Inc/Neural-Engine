# V0.221 — Fused one-token Qwen subset-router audit

**Date:** 2026-09-09  
**Status:** `PARITY-SAFE OPT-IN; DIAGNOSTIC MICRO-OPTIMIZATION`  
**Branch:** `exp/track-runtime`

## Question

The accepted Qwen K=5 route is a 56-way subset router: it predicts one of
the 56 five-group combinations and then activates those five groups. This
probe fuses the two router projections, SiLU, best-subset selection, and
subset-membership mapping into one fixed-shape CUDA kernel. It does not
change circuit weights, correction, active K, or the default PyTorch path.

The probe uses deterministic non-tied final router weights so that the custom
kernel is compared against a stable hard-route contract. It is a routing
parity probe, not a quality or training result.

## Results

Eight Qwen3-0.6B transferred children, float32, E=8/K=5, 20 timing iterations
after 8 warmups:

| Batch | PyTorch subset route | Fused subset route | Fused/PyTorch | Set mismatches | Max child-output error |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.1327 ms | 0.7624 ms | `0.3575x` | `0` | `4.77e-7` |
| 8 | 1.8908 ms | 0.7263 ms | `0.3841x` | `0` | `4.77e-7` |

The route weights matched exactly (`0.0` maximum error). A no-cache full-model
probe with the same deterministic children measured:

| Batch | PyTorch model | Fused model | Fused/PyTorch | Max logit error |
|---:|---:|---:|---:|---:|
| 1 | 40.0389 ms | 32.7268 ms | `0.8174x` | `8.58e-6` |
| 8 | 42.8899 ms | 41.9919 ms | `0.9791x` | `1.43e-5` |

This is a useful diagnostic signal, but the model probe is no-cache and uses
randomized children; it is not sufficient to change the serving default.
The trained-child audit is V0.222.

## Decision

- Keep `single_token_router_backend="torch"` as the default.
- Keep `"cuda-fused-subset"` as an opt-in path pending trained-child and
  production-stream validation.
- Treat the route-stage reduction as a bounded micro-optimization; selected
  FFN/correction dispatch remains the more important runtime target.

## Reproduction

```text
python -u benchmark_qwen_fused_subset_router_audit.py --warmup 8 --iterations 20 --batch-sizes 1 8 --output results/runs/v0_221_fused_subset_router_audit.json
```

## Artifacts

- `benchmark_qwen_fused_subset_router_audit.py`
- `neural_engine/qwen_router_dispatch.py`
- `neural_engine/qwen_router_dispatch.cpp`
- `neural_engine/qwen_router_dispatch.cu`
- integration hook: `benchmark_qwen_multi_layer_transplant.py`
