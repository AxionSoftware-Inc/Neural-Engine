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
True one-token eager decode was still slower (`1.371x/1.403x` for K=5/K=6),
but a fixed-shape CUDA Graph replay smoke on eight layers reached `0.467x`
of the dense parent at K=5 and `0.588x` at K=6, with max eager-logit error
`1.6e-5`. This is a positive opt-in runtime result, not yet a trained
`use_cache` serving claim. Input-buffer replacement inside the fixed shape
also matched eager within `1.1e-5`. The next milestone is a trained K=5 graph
audit with a custom fixed-KV cache, `use_cache` and a small shape cache. The
custom fixed-KV replay now passes `use_cache=True` with K=5 at `0.516x` of the
dense parent and K=6 at `0.595x`, while generic `StaticCache` remains unsafe.
The trained K=5 audit now also passes: repeat `17.391 ms` graph versus
`26.648 ms` dense parent (`0.653x`), with same-seed CE deltas `+0.036528` and
`+0.043563` and alternate-token replay error `5.49e-6`. This closes the trained fixed-shape proof point; prefill,
`generate()`, dynamic shapes and production kernel safety remain open. A
four-step prefill-to-decode replay also passes with maximum error `5.72e-6`,
so the next milestone is now a small generation adapter plus safe fallback for
uncaptured shapes. The opt-in greedy adapter now reproduces an eight-token
sequence exactly between graph and eager paths, and a repeated request reuses
one captured shape entry. Trained-child generation and multi-shape fallback
management remain open; an uncaptured and an evicted shape now have explicit
eager fallback controls that match independent eager runs.
Prefix lengths 4 and 8 also receive separate entries and both pass exact
graph/eager generation parity; batch-size variation and concurrent request
parity; batch-2 now also passes exact replay and measures `0.534x` of eager
generation. Batch 2/4/8 all pass exact replay with graph/eager ratios
`0.534x/0.594x/0.601x`. Batch sizes above 8, trained-child batch quality,
and concurrent request management remain open. With the trained K=5 child,
graph/parent was `0.737x/0.936x` at batch 1/2 but `1.109x/1.445x` at batch
4/8, so larger-batch trained correction still needs a fused kernel.
The single-token BMM specialization trims trained B8 graph/eager from
`1.091x` to `1.034x` without quality or parity regression, but does not yet
beat dense at B8.
V0.204 rank control adds a useful bounded knob: rank 32 passes the quality
gate on seeds 2026/17 and gives trained B8 graph/dense `1.411x/1.344x`, while
rank 16 is `1.479x`. Keep rank 32 opt-in; rank 64 remains the compatibility
default pending longer multi-seed validation.
V0.205 rank 8 is a stronger runtime candidate: with long timing it gives B8
graph/dense `1.390x/1.008x` and graph/sparse-eager `0.924x/0.965x` on seeds
2026/17, with CE deltas `+0.02412/+0.02997`. Keep it opt-in because timing
variance and fused-kernel validation remain open.
V0.206 base-child BMM projection is parity-safe but not a graph speedup:
trained B8 is `35.494 ms` versus einsum `34.974 ms`, so einsum stays default.
The audit now places the known failing packed-capture probe last to preserve
CUDA benchmark state.
V0.207 Inductor fused-child probing is blocked by the local PyTorch/Triton
toolchain (`Cannot find a working triton installation`), not by model parity.
The next runtime experiment must therefore be a static-index/fused correction
path independent of Inductor.
Native's stats-free
serving path already removes diagnostic tensor overhead (`23.5%` faster at
batch-1 in the first smoke); it remains opt-in until a production caller is
wired to it.

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
