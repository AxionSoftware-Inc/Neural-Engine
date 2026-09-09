# V0.212 — Effective-output correction folding audit

**Date:** 2026-09-09  
**Status:** `REJECTED FOR ADOPTION; ROUTE DRIFT`  
**Branch:** `exp/track-runtime`

## Hypothesis

For a selected group, the trained correction is linear in the selected group
output:

```text
selected = coefficient @ W_out.T
correction = selected @ mix_in.T @ mix_out.T
```

Therefore the two operations can be folded into one derived output matrix:

```text
W_eff = W_out + mix_out @ mix_in @ W_out
```

This should remove the correction gather and its two small projection launches
while still evaluating only the selected K groups.

## Local correctness

The derived matrix matches the explicit base-plus-correction formula on the
CPU unit test at `1e-6` tolerance. This confirms the algebra and isolates the
failure below to cascade behavior rather than a simple matrix-orientation bug.

## Trained eight-layer result

Qwen3-0.6B, E=8/K=5, rank 4, layers 19–26, seed 2026, float32, batch 8,
same trained child recipe:

| backend | eager | graph | graph / eager | graph vs own eager | vs vectorized final logits |
|---|---:|---:|---:|---:|---:|
| vectorized | `34.254 ms` | `29.576 ms` | `0.863x` | `9.54e-6` | reference |
| effective-output | `33.301 ms` | `29.050 ms` | `0.872x` | `9.06e-6` | `1.45` |

The effective-output backend is internally graph-stable, but its final output
diverges strongly from the vectorized cascade. The derived projection changes
each layer by only a floating-point-sized amount locally; nevertheless, hard
subset selection is discontinuous, so a later layer can choose a different
subset and amplify the difference. The one-token timing improvement is not
usable without route consistency.

## Decision

- reject all-layer effective-output folding for the normal cascade;
- keep vectorized correction as the default;
- do not interpret the small timing reduction as a speedup;
- only revisit this algebra under frozen-route evaluation or inside a full
  fused layer that preserves the route computation and numerical path.

The custom CUDA correction kernel remains a separate parity-safe opt-in, but it
also did not produce a large end-to-end speed gain in V0.211.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --seed 2026 --warmup 40 --iterations 100 --correction-backend-iterations 30 --single-token-backend-iterations 30 --compiled-child-iterations 2 --batch-sizes 1 8 --output results/runs/v0_212_effective_output_seed2026.json
```

## Artifacts

- opt-in implementation: `benchmark_qwen_multi_layer_transplant.py`;
- cascade A/B harness: `benchmark_qwen_trained_graph_audit.py`;
- local parity test: `tests/test_qwen_packed_dispatch.py`;
- preceding parity-safe CUDA kernel: `RUNTIME_QWEN_CORRECTION_CUDA_KERNEL_AUDIT_20260909.md`.
