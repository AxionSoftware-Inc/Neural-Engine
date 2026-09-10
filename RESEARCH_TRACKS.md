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
| Runtime | Dispatch, memory traffic, fused/compiled kernels used by either track | **Engineering track** | `exp/track-runtime` | Validate sequence/production shapes for native fused factorized dispatch; dynamic routing remains graph-unsafe |

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
V0.208 no-correction ablation fails the quality gate (`+0.091023`) despite
near-dense B8 graph runtime (`1.047x`), confirming that correction capacity is
necessary for the current K=5 quality point. Rank 8 remains the smallest
viable tested candidate. A three-seed rank-4 follow-up passes the quality gate
(`+0.03733/+0.04004/+0.04086`) and gives B8 graph/dense
`1.339x/1.330x/1.337x`; rank 4 is now the smallest tested viable opt-in.
Longer-budget validation is still needed before it can replace rank 64 as the
compatibility default.
V0.210 doubles child/hard/router steps for rank 4; seeds 2026/17 still pass
(`+0.02795/+0.03825`) and B8 graph/dense improves to `1.280x/1.265x`.
This removes short-budget fragility as the leading explanation, but the fused
correction kernel remains the real latency target.
V0.211 adds a fixed-shape custom CUDA correction kernel. After binding it to
the current stream, trained final-logit parity is `9.3e-6/9.5e-6` versus the
vectorized backend; graph replay is safe. End-to-end graph timing changes only
`49.36→48.88 ms` in the measured seed, so the kernel is retained opt-in and
not promoted as a large speed breakthrough.
V0.212 tests folding the linear correction into an effective output matrix
(`W_eff = W_out + mix_out·mix_in·W_out`). It reduces one-token backend time
slightly, but all-layer cascade final-logit parity versus vectorized is `1.45`,
so route drift makes it unsafe. The algebra is retained as a rejected opt-in;
the next fusion must preserve routing decisions or fuse the full layer.
V0.213 reruns the rank-4 long-budget recipe on a third seed (42). It also
passes the quality gate (`+0.040236`) with B8 graph/dense `1.234x`,
graph/sparse-eager `0.871x`, and final-logit parity `8.34e-6`; generation
remains exact. Rank 4 therefore has three-seed long-budget support as an
opt-in, but rank 64 remains the compatibility default and correction dispatch
is still the main runtime target.
V0.214 fuses selected Qwen base output and low-rank correction in one fixed-
shape CUDA dispatch. Two seeds give `0.842x/0.859x` graph time versus the
vectorized correction backend, with final-logit error `1.62e-5/1.67e-5`;
the full sparse graph reaches `0.957x/1.000x` of the dense parent. This is an
`ACCEPTED OPT-IN` correction backend, not a default switch or a claim that
the whole Qwen serving path is faster than dense.
V0.215 sweeps that backend at batch 1 and 8. B8 remains faster in both seeds,
but batch-1 ratios are inconsistent (`1.276x` and `0.720x`), so no automatic
batch policy is accepted. Vectorized remains the default; repeated interleaved
timing and fused router/top-k dispatch are the next runtime targets.
V0.217 performs that interleaved timing on two seeds. Batch-1 fused/vectorized
ratios are `1.478x/1.468x`; B8 ratios are only `0.978x/0.975x`, with final-logit
error below `1.6e-5`. The fused-full implementation is parity-safe but its
large speed claim is rejected; the default stays vectorized and the next real
target is router/top-k/dispatch fusion.
V0.218 tests a one-block-per-token correction kernel intended to remove
atomic accumulation. It is parity-safe (`9.5e-6/1.6e-5`) but slower than
vectorized by `4.661x` at B1 and `1.726x` at B8, so the hypothesis is rejected;
the loss is serialized FFN work, not atomics.
V0.219 profiles the child stages directly: at B1 router+top-k is `1.391 ms`
versus `2.553 ms` selected FFN dispatch; at B8 they are `1.008 ms` versus
`17.925 ms`. V0.220 fuses the router projections, top-k and softmax. It is
parity-safe on non-tied routes and improves the route stage to `0.636x/0.675x`
at B1/B8, but child end-to-end is only `0.913x/1.024x`; B32 regresses to
`1.014x`. Keep it opt-in and unchanged by default. This confirms router
fusion is a bounded small-batch optimization, not the main dispatch solution.
V0.221 applies the same idea to the real 56-way K=5 subset router; its
non-trained probe reaches `0.358x/0.384x` route-stage time at B1/B8 with
zero set mismatch. V0.222 repeats it on the accepted trained K=5 recipe:
quality remains `+0.042135` CE, eager fused/PyTorch is `0.980x/1.020x`, and
graph fused/PyTorch is `1.032x/1.011x` at B1/B8. An eight-token CUDA-Graph
greedy generation also matches exactly. A current-stream bug found during
graph replay was fixed and the rerun is parity-safe (`<=1.07e-5`
fused-vs-PyTorch eager logit error). Keep it opt-in; graph serving sees no
speed breakthrough, so selected FFN dispatch remains the active runtime
target.
V0.223 fixes the grouped selected-FFN path's CUDA Graph blocker by replacing
capture-time `bincount`/host-shape logic with a graph-stable token-count upper
bound. On trained K=5 seeds 2026/2027, grouped-vs-single-token graph time is
`0.963x/0.953x` at B1 and `0.589x/0.589x` at B8; a B32 extension reaches
`0.423x`. It has exact eight-token generation parity and max logit error below
`1.4e-5`. This is the first strong runtime result aimed at the selected FFN
bottleneck. Prefix lengths 4/32/128 at B8 remain `0.593x/0.600x/0.663x`,
so the signal persists with context even though attention dilutes it; broader
production-shape validation remains open. Against the restored dense parent,
grouped sparse graph is still `1.099x/1.146x/1.259x` at B1/B8/B32, so this is a
strong sparse-path improvement but not yet a dense-serving win.
V0.224 checks the `grouped-fused` one-BMM variant: it is parity-safe, but only
`1.000x/0.993x/0.997x` of ordinary grouped graph time at B1/B8/B32. It helps
eager B1/B8 and is neutral at B32, while the dense gap remains; no batch policy
or default switch is added.
V0.229 extends the correction-rank sweep to a longer `600/600/200` trained
recipe at rank 1. Seeds 2026/17 pass the quality gate with CE deltas
`+0.045878/+0.035236`; B8 grouped-fused graph/dense is `1.048x/1.044x`, and
both eight-token generation checks are exact. Rank 1 is therefore the smallest
tested viable opt-in candidate, but the dense-serving gap remains open and the
rank-64 compatibility default is unchanged. The next runtime target is
selected-FFN launch/packing fusion, not further router work.
V0.230 checks the same rank-1 recipe at B1/B8/B32 on both seeds. The
grouped-fused graph/dense ratios are `1.080x/1.044x/1.065x` and
`1.078x/1.042x/1.060x`; quality deltas remain within `+0.05` and generation
parity is exact. This confirms shape robustness of the opt-in candidate but no
dense-serving win. Grouped-fused is not promoted over ordinary grouped; the
next target remains selected-FFN launch/packing fusion.
V0.231 caches route-independent grouped pair metadata by fixed shape. Across
two seeds and B1/B8/B32, cached/grouped graph ratios range from `0.994x` to
`1.002x`; exact generation and logit parity remain intact. This is retained as
a safe opt-in micro-optimization, not a speed claim. Metadata reuse is no
longer the priority; selected-FFN packing/projection/scatter fusion is.
V0.232 folds the rank-1 cross-group correction into grouped selected-output
accumulation, avoiding the wrapper-side selected-output reorder and second
correction pass. Across two seeds and B1/B8/B32 it is `0.991x–0.997x` of
ordinary grouped graph time, with exact generation parity and quality deltas
`+0.045226/+0.040347`. The gain is consistent but small; the dense gap remains
`1.037x–1.081x`, so this stays opt-in and the next target is deeper tiled/full
selected-FFN fusion.
V0.233 removes the uniform K-subset route weight multiply under an explicit
subset-router contract. Across two seeds and B1/B8/B32 it is only `0.995x–1.002x`
of the fused-correction path, with exact parity but no material or consistent
gain. The shortcut is rejected for adoption and retained only as a diagnostic;
selected-FFN tiled/full fusion remains the next target.
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
