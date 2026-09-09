# V0.202 — Trained rank-64 correction backend A/B

**Date:** 2026-09-09  
**Status:** `VECTORIZED ACCEPTED OPT-IN; PACKED GRAPH REJECTED`  
**Branch:** `exp/track-runtime`

## Question

The trained batch audit showed that graph replay helps at batch 1/2 but loses
most of its advantage at batch 8. The suspected cost center is the rank-64
cross-group correction. This A/B compares its vectorized gather path with the
V0.193 packed accumulation path, without changing child weights or routing.

## Setup

- same trained K=5 cascade as V0.197;
- eight replaced layers `19–26`, batch 8, one decode token after a four-token
  prefix;
- 30 warmup/timing iterations per backend;
- vectorized backend uses the existing 128 MiB gather guard;
- packed backend is forced with `max_dense_gather_bytes=0`.

## Result

| correction backend | eager | graph | graph / eager | parity | decision |
|---|---:|---:|---:|---|---|
| vectorized gather | `41.630 ms` | `45.400 ms` | `1.091x` | `1.00e-5` | retain |
| packed accumulation | `79.794 ms` | capture failed | — | — | reject for graph |

The vectorized path remains graph-safe and numerically matches eager within
`1.01e-5`, but at trained batch 8 it is slightly slower than eager. The
packed path is slower even in eager mode for this decode shape and cannot be
captured because its per-expert loop calls `torch.where` on device routing
indices while the CUDA stream is being captured (`operation not permitted
when stream is capturing`).

## Decision

Do not switch the trained graph path to packed correction. Keep the
vectorized path for its correctness and graph safety. The large-batch trained
runtime bottleneck is now localized to correction projection/gather work; the
next engineering direction is a static-index or fused correction kernel that
does not call device-dependent `where`/allocation operations during capture.

This is a runtime backend result, not evidence against the sparse circuit
quality recipe. The model quality remains the V0.197 result (`+0.03653` to
`+0.04356` same-seed repeats, both under the gate).

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --iterations 30 --correction-backend-iterations 30
```

## Artifact

- benchmark and backend audit: `benchmark_qwen_trained_graph_audit.py`;
- correction implementation: `benchmark_qwen_multi_layer_transplant.py`.
