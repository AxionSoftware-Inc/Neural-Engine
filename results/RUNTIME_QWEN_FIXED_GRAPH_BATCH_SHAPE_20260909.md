# V0.201 — Fixed-graph batched decode audit

**Date:** 2026-09-09  
**Status:** `POSITIVE OPT-IN; BATCH-2 PARITY PASS`  
**Branch:** `exp/track-runtime`

## Purpose

The original fast-path condition only selected the gathered batched
contractions when the flattened token count was exactly one. That unnecessarily
sent batch-2 one-token decode through `bincount`, which is unsafe inside a
CUDA Graph. The selected-group contraction already supports multiple batch
rows, so the guard was changed to detect a single sequence-token dimension.

## Setup

- local `Qwen/Qwen3-0.6B`, float32 CUDA;
- layers `19–26`, E=8/K=5 copied sparse children;
- batch sizes 2, 4, and 8, four-token prefix, eight generated tokens;
- one graph capture followed by a same-shape reuse;
- independent eager custom-cache generation for comparison;
- 2 warmup and 10 timed reuse iterations.

## Result

| batch | graph reuse | eager | graph / eager | parity / reuse |
|---:|---:|---:|---:|---|
| 2 | `156.52 ms` | `293.34 ms` | `0.534x` | exact / exact |
| 4 | `189.64 ms` | `319.06 ms` | `0.594x` | exact / exact |
| 8 | `261.79 ms` | `435.81 ms` | `0.601x` | exact / exact |

Each batch size used one graph capture and thirteen same-shape cache hits in
the timed run. Overall status: `PARITY_PASS`.

The batch-2/4/8 paths no longer reach `bincount` during capture. They are
numerically exact at the token level and substantially faster than the
repeated eager generation loop in these measurements.

## Decision and next step

`V0.201` is accepted as an opt-in batch-2/4/8 fixed-shape runtime result. The
default model remains unchanged. Batch sizes above 8, concurrent request
isolation, trained-child batch quality/runtime, and production stream safety
remain open.

## Reproduction

```text
python -u benchmark_qwen_fixed_graph_batch_shape.py --local-files-only --batch-size 2 --iterations 10
python -u benchmark_qwen_fixed_graph_batch_shape.py --local-files-only --batch-size 4 --iterations 10
python -u benchmark_qwen_fixed_graph_batch_shape.py --local-files-only --batch-size 8 --iterations 10
```

## Artifacts

- fast-path implementation: `benchmark_qwen_multi_layer_transplant.py`;
- batch benchmark: `benchmark_qwen_fixed_graph_batch_shape.py`;
- shape pool/decoder: `neural_engine/qwen_fixed_graph.py`.
