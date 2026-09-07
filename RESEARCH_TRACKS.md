# Research tracks and branch map

This file is the navigation layer for the repository. Historical experiments
remain in `results/`; this map defines which work is active and which results
can be compared.

## Current decision

The repository contains three related but independent tracks. They share
benchmarking and runtime utilities, but they are not one model and their
quality numbers must not be compared as if they were the same task.

| Track | Scope | Current status | Working branch | Primary next step |
|---|---|---|---|---|
| Native Engine | Original attention-free Neural Engine on synthetic algorithmic tasks | **Validated local baseline** | `exp/track-native-engine` | Improve small-batch and decode runtime without changing the model body |
| Sparse Qwen | Exact Qwen FFN/SwiGLU transfer into sparse circuits and learned subset routing | **Validated at K=5/K=6; K=4 open** | `exp/track-qwen-sparse` | Work only on learned candidate retrieval/subset regret |
| Runtime | Dispatch, memory traffic, fused/compiled kernels used by either track | **Engineering track** | `exp/track-runtime` | Benchmark and implement a compiled decode path |

`freeze/baseline-20260907` is the read-only starting point for this
organization pass. The existing `main` and `freeze/v0.27-audited` branches are
preserved. No historical branch or result is deleted.

## Baselines

### Native Engine V0.12

This is the current product-like reference for the original architecture. It
is a synthetic numeric/composition benchmark, not a language-model result.

- Validation accuracy: `71.98%` (`51.02%` for the trained dense reference).
- RTX 3060, batch 128: `8.322 ms`, `15,382 samples/s`.
- Estimated unique active parameters: `1.977M / 20.247M` (`9.77%`).
- Current limitation: small-batch and one-token latency can be dominated by
  routing and kernel-launch overhead.
- Source report: `results/CHECKPOINTED_INFERENCE.md`.

### Sparse Qwen transfer

The exact transfer control converts all 28 Qwen-0.6B SwiGLU MLP layers with
zero reconstruction error. This validates the FFN-to-circuit representation;
it does not validate replacing the whole Transformer, including attention.

- K=6 (`75%` of the eight Qwen experts active): two-seed CE deltas
  `+0.01141/+0.01837`, quality gate passed.
- K=5 (`62.5%` active): `+0.03881/+0.04036`, accepted as the lower-budget
  operating point.
- K=4 (`50%` active): learned routing fails at `+0.06462/+0.06165`, while
  exact paired-subset oracle passes at `+0.01607/+0.01227`.
- Interpretation: the current K=4 limitation is primarily learned routing and
  subset regret, not proof that useful sparse subsets do not exist.
- Source reports: `results/V0_174_CORRECTED_ROUTING_AND_EXPERT_AUDIT.md` and
  `results/V0_193_QWEN_CORRECTION_DISPATCH_AUDIT.md`.

### Runtime baseline

The Qwen selected-token correction dispatch was repaired in V0.193. The old
hard-path gather measured about `2.135x` dense; the comparable smoke measured
`1.128x`, and the trained two-seed timing was approximately `0.99x/1.00x`.
True one-token decode is still slower (`1.371x/1.403x` for K=5/K=6), so the
next runtime milestone is a compiled decode kernel rather than another router
rewrite. Native's stats-free serving path already removes diagnostic tensor
overhead (`23.5%` faster at batch-1 in the first smoke); it remains opt-in
until a production caller is wired to it.

## Problem ownership

The existing `problems.md` is the single problem registry. P-001–P-007 and
the historical C-* records cover Native Engine work; `QWEN-001` and
`RUNTIME-001` explicitly identify the two other tracks. This keeps one place
for expert handoffs without mixing their benchmarks or acceptance gates.

## Branch rules

1. One hypothesis per branch and one primary variable per benchmark.
2. Begin as an opt-in configuration; do not silently change the default model.
3. Native changes go only to `exp/track-native-engine`.
4. Qwen transfer/routing changes go only to `exp/track-qwen-sparse`.
5. Kernel, dispatch, memory-layout and decode changes go only to
   `exp/track-runtime`.
6. A shared utility change is cherry-picked into the affected track branches
   with a separate commit; it is not developed by editing several branches at
   once.
7. Every completed run gets a report under `results/`, a reproducible command,
   seed information, and an explicit `ACCEPTED`, `REJECTED`, or `OPEN` result.
8. A rejected experiment is never removed. It is linked from the relevant
   problem or result index.

## Working order

1. Keep Native V0.12 and Sparse Qwen K5/K6 frozen as baselines.
2. Run runtime work independently because it can benefit both tracks.
3. Continue Native Engine only with small-batch/decode measurements or a
   clearly defined quality/generalization experiment.
4. Continue Qwen only on QWEN-001; do not revisit already rejected compact
   basis, pairwise-router, or naive shared-residual variants without new
   evidence.
5. Integrate a track into `main` only after a two-seed benchmark passes its
   stated gate and the result is documented.
