# V0.218 — Token-block correction dispatch

**Date:** 2026-09-09  
**Status:** `REJECTED FOR SPEED; PARITY RETAINED FOR AUDIT`  
**Branch:** `exp/track-runtime`

## Hypothesis

The previous fused-full kernel launches one block per selected `(token, group)`
pair and uses atomic accumulation for the final hidden row. This experiment
uses one block per token, visits all K=5 selected groups inside that block,
and writes the final row once. It tests whether removing atomics and reducing
block count improves dispatch latency.

## Results

The interleaved timings alternate vectorized, pair-block fused-full, and the
new token-block kernel on one CUDA stream. Ratios below are token-block versus
vectorized.

| seed | batch | vectorized | pair-block fused-full | token-block fused | token/vectorized | max token logit error |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 1 | `14.483 ms` | `20.817 ms` | `67.502 ms` | `4.661x` | `9.54e-6` |
| 42 | 8 | `35.141 ms` | `33.282 ms` | `60.657 ms` | `1.726x` | `1.62e-5` |

The kernel is numerically safe in this fixed-shape float32 test, but it is
substantially slower. Serializing K group FFN projections inside one block
loses the batched GEMM parallelism that the vectorized path uses; atomics were
not the dominant cost.

## Decision

- Reject token-block dispatch for adoption and do not use it as a default or
  automatic policy.
- Keep the source only as a recorded negative experiment; the existing
  vectorized backend remains default.
- Do not continue optimizing this one-block-per-token shape without a new
  layout or tensor-core design.
- Move the runtime effort to router/top-k/index fusion or a genuinely batched
  selected-GEMM kernel.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --child-steps 300 --hard-steps 300 --router-steps 100 --seed 42 --warmup 15 --iterations 50 --correction-backend-iterations 8 --single-token-backend-iterations 15 --compiled-child-iterations 1 --interleaved-timing-iterations 20 --batch-sizes 1 8 --output results/runs/v0_218_token_block_correction_seed42.json
```

## Validation

The run reports `PARITY_PASS`; repository tests remain `158 passed, 2
warnings`. The packed graph-capture failure is still intentionally probed
last.
