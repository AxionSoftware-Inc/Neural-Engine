# V0.198 — Custom fixed-KV multi-step prefill-to-decode replay

**Date:** 2026-09-09  
**Status:** `POSITIVE OPT-IN; MULTI-STEP PARITY PASS`  
**Branch:** `exp/track-runtime`

## Purpose

V0.197 proved the custom fixed-KV graph with a trained child at one decode
position. This smaller runtime control checks the stateful contract needed by
generation: after one prefix prefill, the same fixed-shape graph must accept a
new token and advance the KV write position on every replay.

## Setup

- local `Qwen/Qwen3-0.6B`, float32 CUDA;
- layers `19–26`, E=8/K=5 sparse transferred children;
- four-token prefix, followed by four one-token decode replays;
- one captured graph shape `[batch=1, token=1]`;
- both the token input buffer and the fixed KV decode position are updated
  between replays;
- independent eager run uses a fresh custom cache and the same positions.

This is a runtime state/parity control. The children are copied for this
smoke, not trained; trained quality is covered by V0.197.

## Result

| decode step | max graph-vs-eager logit error |
|---:|---:|
| 0 | `5.25e-6` |
| 1 | `5.72e-6` |
| 2 | `5.25e-6` |
| 3 | `5.25e-6` |

Overall maximum: `5.72e-6`; status: `PARITY_PASS` at the `1e-3` threshold.

The custom cache therefore supports a prefix prefill followed by multiple
decode positions without replaying stale KV state or writing every token to
the same slot. This is the core state transition required by a generation
adapter.

## Decision and next step

`V0.198` is accepted as an opt-in fixed-shape runtime result. It closes the
multi-step cache-state question for the tested shape and four-step sequence.
It does not yet provide a `generate()` API, dynamic sequence-shape handling,
shape-cache eviction policy, or a production kernel/stream guarantee. The
next runtime task is a small generation adapter that reuses this cache and
falls back safely when a requested shape is not captured.

## Reproduction

```text
python -u benchmark_qwen_custom_kv_multistep.py --local-files-only
```

## Artifact

- code: `benchmark_qwen_custom_kv_multistep.py`.
