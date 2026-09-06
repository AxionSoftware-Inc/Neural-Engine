# Expert feedback: pairwise cost-router experiment

This is a handoff note for the author of the pairwise cost-router proposal.
It records exactly what was implemented, what was measured, and which parts
were not yet tested. All deltas below are held-out cross-entropy deltas versus
the dense Qwen3-0.6B parent; lower is better. The project quality gate is
`+0.05`.

## What was implemented

The proposed 70-subset output head was replaced with a structured 36-output
head for `E=8`, `K=4`:

- 8 individual group-cost outputs;
- 28 pair-cost outputs;
- each of the 70 K4 subset costs is reconstructed as the sum of four singles
  and six pair terms;
- inference still executes only the selected four groups;
- final corrected group outputs were used to build the teacher subset costs;
- the existing subset-soft objective was tested;
- a direct expected-regret objective was added, with normalized regret
  (`MSE / hidden_size`) plus a small auxiliary soft-target cross-entropy term.

The prior pairwise run was not a final-checkpoint router-only test. Its actual
order was 100 router-supervision steps, then 300 child/correction hard-training
steps while the router was frozen, followed by zero post-router refit steps.
The copied experts were fixed, but the low-rank cross-group correction changed
after the initial router fit. The controls used Qwen3-0.6B, float32 CUDA,
rank-64 cross-group correction, matched K4 scale (`scale=4`), and the same
held-out benchmark used by the preceding routing audit.

## Results

| depth | router/objective | learned CE delta | paired exact-oracle delta | timing / dense parent | decision |
|---|---|---:|---:|---:|---|
| 2 layers, 25–26 | pairwise cost + old subset-soft | `+0.05427` | `+0.02871` | `1.166x` | no improvement |
| 2 layers, 25–26 | pairwise cost + normalized regret | `+0.04996` | `+0.03033` | `1.163x` | marginal gate pass |
| 4 layers, 23–26 | pairwise cost + normalized regret | `+0.05206` | `+0.02904` | `1.328x` | gate fail |

The old hidden-state subset-router controls were approximately `+0.05445`
(learned) and `+0.03282` (oracle) at two layers, and `+0.04038` and
`+0.02273` at four layers. Therefore the pairwise head did not produce a
reliable improvement over the existing representation. The two-layer
normalized-regret result is close to the gate, but the four-layer result does
not preserve it.

An unnormalized-regret trial was also run. It reached approximately `+0.06130`
learned and `+0.03653` oracle and was superseded by the normalized version;
the scale of the regret target was not stable enough for that formulation.

## Interpretation

The pairwise interaction hypothesis is plausible and the implementation is
mathematically aligned with the additive corrected output. However, in this
recipe it is not a solution to the depth/cascade router gap:

1. The exact oracle remains good at four layers (`+0.02904`), so useful
   subsets still exist.
2. The learned router remains just outside the gate (`+0.05206`), so the main
   remaining loss is route generalization/cascade distribution shift.
3. The pairwise representation was not expanded to eight layers because the
   four-layer control did not justify the additional compute.
4. The pairwise code is retained as an optional research path, not promoted to
   the default architecture.

## Expert review and clean refit control

The proposal author reviewed the JSON and corrected the protocol reading: the
original pairwise run used 100 router steps, then 300 correction/child hard
steps, with no post-router refit. This means the router was trained before the
correction reached its final state. The recommended control was a static
final-target refit versus iterative cascade data aggregation.

We implemented and ran that control on the same four-layer K4 setup. Both
variants used the same initial 100 router steps and 300 child/correction steps;
only the final router refit differed:

| seed | final refit | learned CE delta | mean layer regret | mean layer p95 regret | gate |
|---|---|---:|---:|---:|---|
| 2026 | static, 300 steps | `+0.04712` | `0.10572` | `0.42709` | pass |
| 2026 | aggregate, 3×100 steps | `+0.04781` | `0.10422` | `0.42693` | pass |
| 2027 | static, 300 steps | `+0.04120` | `0.09981` | `0.41281` | pass |
| 2027 | aggregate, 3×100 steps | `+0.04220` | `0.10041` | `0.41379` | pass |

Aggregation did not beat static by the proposed `0.005` CE on either seed: it
was `0.00069` worse on seed 2026 and `0.00100` worse on seed 2027. Mean regret
reduction was only about 3.6–5.8%, below the proposed 20% criterion. Global
p95 regret improved, but a per-layer tail regression remained around layer 24
on seed 2026. The evidence supports static final-target refit as a useful
protocol correction, but does not justify the extra iterative aggregation or
expansion to eight layers.

## Third-seed and parameterization follow-up

The static recipe was then checked on seed 2028. It reached learned
`+0.05705` while the paired oracle remained good at `+0.02749`, so the
two-seed pass is not yet stable. Two targeted variants were also tested on the
same seed:

| variant | learned CE delta | oracle CE delta | decision |
|---|---:|---:|---|
| ordinary static final refit | `+0.05705` | `+0.02749` | fail |
| top-25% tail-regret refit | `+0.05574` | `+0.02745` | fail; small gain |
| 27-D centered-basis refit | `+0.05366` | `+0.03177` | fail; best of these, still over gate |

The centered basis removes the eight-dimensional non-identifiability in the
36 component coordinates and improves this seed by about `0.0034`, but does
not restore the quality gate. Tail-aware regret improves the result by only
about `0.0013`. In all cases the oracle remains substantially better, so the
remaining issue is learned route/cascade generalization rather than missing
subset capacity.

## Related scale controls

The scale question was also corrected before evaluating the router. The sparse
sum must use `scale=K`, so that the coefficient is `scale/K=1`, as in the K4
and K6 references. With matched scale, K5 passed on two seeds:

- learned `+0.04101` and `+0.04186`;
- paired oracle `+0.00512` and `+0.00055`;
- 62.5% of groups active, about `1.91x` current Python/PyTorch timing.

K6 also passed on two seeds (`+0.01103`, `+0.02117`) at 75% active. A learned
amplitude/scale predictor was not promoted: optimal-scalar diagnostics gave
mean `g*≈0.992` and only `0.00158` K5 local-MSE gain. This indicates that the
main issue is not output amplitude but route selection and cascade alignment.

## Remaining follow-up questions

The proposed static-refit versus aggregation control is now complete. The
remaining expert feedback that would most affect the next decision is:

- Should the 20% regret criterion be applied per layer or to the global
  token-weighted cascade, given that global p95 improved but layer 24 has a
  tail regression on seed 2026?
- Is the small two-seed CE pass from static final-target refit enough to call
  pairwise cost routing viable, or should it receive a third seed/independent
  text validation before being retained?
- Should the next architectural test use the proposed 27-dimensional
  centered subset-cost basis, or is a tail-aware objective more justified?

## Reproducibility artifacts

The implementation is in `benchmark_qwen_multi_layer_transplant.py`. The
individual run JSON files are local under `results/runs/` and are ignored by
Git; the aggregate numbers above are recorded in
`V0_174_CORRECTED_ROUTING_AND_EXPERT_AUDIT.md`.

The code, p95 diagnostic, refit controls, and this handoff note are pushed on
branch `exp/scale-invariant-routing`; the latest commit is recorded after the
next documentation commit.
