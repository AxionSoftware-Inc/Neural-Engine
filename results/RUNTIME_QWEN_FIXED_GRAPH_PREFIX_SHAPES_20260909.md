# V0.200 — Fixed-graph prefix-shape audit

**Date:** 2026-09-09  
**Status:** `POSITIVE OPT-IN; PREFIX-SHAPE PARITY PASS`  
**Branch:** `exp/track-runtime`

## Purpose

The shape pool was already tested with different decode budgets. This control
checks that prefix length is also part of the graph key: a graph captured for
one prefix length must not be reused with another prefix/cache shape.

## Setup

- local `Qwen/Qwen3-0.6B`, float32 CUDA;
- layers `19–26`, E=8/K=5 copied sparse children;
- one-token graph decode, eight generated tokens;
- prefix lengths 4 and 8, both batch 1;
- bounded shape pool with three entries;
- each graph result compared with an independent eager custom-cache run;
- the four-token shape was requested a second time to test reuse.

## Result

| check | result |
|---|---|
| 4-token prefix graph vs eager | **exact** |
| 8-token prefix graph vs eager | **exact** |
| reused 4-token prefix graph vs eager | **exact** |
| graph captures / cache hits | `2 / 1` |
| status | `PARITY_PASS` |

The pool captured separate entries for prefix lengths 4 and 8, then reused
the original four-token entry without a new capture. No stale KV state or
wrong prefix shape was observed.

## Decision and next step

`V0.200` is accepted as an opt-in prefix-shape control. It does not claim
arbitrary dynamic batching or production `generate()` support. Batch-size
variation, multiple concurrent requests, graph memory pressure, and the
trained-child longer-prompt audit remain open.

## Reproduction

```text
python -u benchmark_qwen_fixed_graph_prefix_shapes.py --local-files-only
```

## Artifact

- code: `benchmark_qwen_fixed_graph_prefix_shapes.py`;
- pool: `neural_engine/qwen_fixed_graph.py`.
