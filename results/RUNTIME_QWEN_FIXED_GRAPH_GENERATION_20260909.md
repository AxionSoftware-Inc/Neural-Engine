# V0.199 — Fixed-shape greedy generation adapter

**Date:** 2026-09-09  
**Status:** `POSITIVE OPT-IN; GRAPH/EAGER GENERATION PARITY PASS`  
**Branch:** `exp/track-runtime`

## Purpose

V0.198 verified multi-step cache state. This follow-up wraps that contract in
a small greedy-generation helper: prefix prefill produces the first token,
then the fixed one-token CUDA Graph is replayed while the token input buffer
and KV write position advance. The helper also exposes an eager path for
uncaptured or unsupported shapes.

## Smoke setup

- local `Qwen/Qwen3-0.6B`, float32 CUDA;
- layers `19–26`, E=8/K=5 copied sparse children;
- four-token prefix and eight generated tokens;
- graph and eager paths use independent custom fixed-KV caches;
- greedy `argmax` decoding, no sampling;
- the children are copied for this runtime smoke, not trained. Trained
  quality and graph parity are covered by V0.197.

## Result

| check | result |
|---|---|
| generated token count | `8` |
| graph vs eager exact token sequence | **true** |
| status | `PARITY_PASS` |

The graph and eager adapters generated the same complete token sequence. This
confirms the integration path does not reuse stale token inputs or the wrong
KV position as the sequence advances.

## Decision and next step

`V0.199` is accepted as an opt-in fixed-shape generation adapter. It does not
change the default Qwen model or Hugging Face `generate()` behavior. The next
work is to add a shape-keyed capture cache and a safe fallback policy, then
repeat the adapter test with the trained K=5 child and longer prompts.

## Reproduction

```text
python -u benchmark_qwen_fixed_graph_generation.py --local-files-only
```

## Artifact

- adapter: `neural_engine/qwen_fixed_graph.py`;
- smoke: `benchmark_qwen_fixed_graph_generation.py`.
