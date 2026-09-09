# V0.203 — Single-token correction BMM dispatch audit

**Date:** 2026-09-09  
**Status:** `SMALL POSITIVE OPTIMIZATION; QUALITY/PARITY PRESERVED`  
**Branch:** `exp/track-runtime`

## Hypothesis

For one sequence token, the rank-64 correction’s generic ellipsis `einsum`
may dispatch less efficiently than explicit batched matrix multiplies. The
correction formula and gathered weights remain unchanged; only the decode
contraction implementation is specialized for the single-token shape.

## Quality control

The accepted trained K=5 recipe was rerun on the same seed. Teacher CE was
`4.785308`; sparse CE delta was `+0.035745`, below the `+0.05` gate. This is
within the prior same-seed range (`+0.036528`, `+0.043563`, `+0.038338`) and
shows no quality regression.

## Trained B8 backend result

| backend | eager | graph | graph / eager | max parity error |
|---|---:|---:|---:|---:|
| vectorized BMM | `43.286 ms` | `44.748 ms` | `1.034x` | `8.94e-6` |
| packed accumulation | `84.870 ms` | capture failed | — | — |

The previous vectorized correction implementation measured `1.091x` graph /
eager in the comparable B8 audit. The BMM specialization reduces that ratio
to `1.034x`, a small but measurable directionally positive change. It does
not yet make the trained B8 graph faster than dense parent; the batch matrix
from the same run measured graph/parent `1.491x` at B8.

The packed backend remains rejected for graph use because its per-expert
`torch.where` over device routing indices is not permitted during capture.

## Decision and next step

Keep the single-token BMM specialization as the current graph-safe correction
path. It is an opt-in implementation improvement, not a new model default or
a quality breakthrough. The next runtime target is a fused/static-index
correction kernel that avoids gathered rank-64 weight traffic and remains
capture-safe at larger batches.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --iterations 30 --correction-backend-iterations 30
```

## Artifacts

- BMM specialization: `benchmark_qwen_multi_layer_transplant.py`;
- trained audit: `benchmark_qwen_trained_graph_audit.py`;
- previous backend control: `results/RUNTIME_QWEN_TRAINED_CORRECTION_BACKEND_AUDIT_20260909.md`.
