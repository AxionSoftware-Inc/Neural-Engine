# V0.254 — Route-count bucketed grouped-GEMM audit

**Date:** 2026-09-09  
**Status:** `REJECTED; EAGER SLOWER AND GRAPH HAS NO GAIN`  
**Branch:** `exp/track-runtime`

## Question

The uniform grouped BMM computes `max_count` rows for every expert, including
padding. This probe groups experts with equal route counts and runs compact
per-bucket BMMs, intending to reduce padded arithmetic.

The route counts are data-dependent. Therefore CUDA Graph capture safely falls
back to the normal uniform grouped projection; the bucketed path is an eager
feasibility probe, not a graph-safe serving backend.

## Protocol

Local `Qwen/Qwen3-0.6B`, layers `19--26`, `E=8`, `K=5`, rank-1 correction,
float32 CUDA, and trained-cascade recipe `600/600/200`. Prefix length is 4;
batch sizes are 1, 8, and 32. The screen uses 5 CUDA-Graph warmups and 10
timing iterations at each shape, with seeds 2026 and 17. The control is the
existing `grouped-adaptive` path.

## Latency

Eager and CUDA-Graph values are milliseconds; lower is better.

| seed | path | B1 eager/graph | B8 eager/graph | B32 eager/graph |
|---:|---|---:|---:|---:|
| 2026 | adaptive | 37.753 / 11.272 | 36.517 / 16.978 | 45.676 / 34.493 |
| 2026 | bucketed | 38.720 / 11.234 | 42.552 / 16.960 | 47.050 / 34.436 |
| 17 | adaptive | 37.611 / 11.245 | 39.080 / 16.904 | 44.775 / 34.544 |
| 17 | bucketed | 39.089 / 11.175 | 39.676 / 17.012 | 47.389 / 34.474 |
| **two-seed eager change** | **bucketed vs adaptive** | **+3.25%** | **+8.77%** | **+4.41%** |

The additional count synchronization, expert indexing, extra BMM launches,
and compacting copies cost more than the saved padded work. The graph path
does not use the bucketed projection because its shapes cannot be captured
without data-dependent allocation/metadata.

## Correctness and quality

- Both seeds passed the strict graph/eager numerical gate; maximum errors were
  about `1.2e-5` against the `1e-3` tolerance.
- Eight-token greedy generation matched the grouped baseline exactly.
- The inference-only backend does not change the trained model or quality;
  CE deltas versus dense were `+0.044466` and `+0.042877`.

Raw records:

- `runs/v0_254_bucketed_grouped_projection_seed2026.json`
- `runs/v0_254_bucketed_grouped_projection_seed17.json`

## Decision

Reject this bucketed implementation as a serving path. It is slower in eager
mode and produces no graph-mode gain. A useful route-count-aware backend must
move the variable-shape scheduling into a fused/autotuned CUDA implementation
rather than paying the current Python/PyTorch synchronization and launch
overhead.
