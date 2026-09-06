# V0.174 — Corrected operator parity and routing audit

Date: 2026-09-06  
Branch: `exp/scale-invariant-routing`

This audit follows `KEYINGI_YOL_2026-09-06.md`. It is the first follow-up
after making hard training and grouped inference use the same explicit route
scale and after freezing copied Qwen experts and routers during correction
refinement.

## Implementation changes

- Hard token-loop training now uses the same `hard_route_scale` as grouped
  inference. The previous implicit `E/K` path could train one operator and
  evaluate another.
- Copied Qwen gate/up/down slices and routers are frozen during joint,
  layerwise, and block correction refinement. Only correction parameters are
  optimized, so cached grouped buffers cannot become stale relative to the
  trainable copied weights.
- Added a two-layer hard-route block-distillation control. It was restricted
  to two consecutive layers and normalized both local and block losses.
- Added leave-one-child ablation, paired exact-oracle routing evaluation, and
  held-out routing diagnostics (`exact_subset_match`, oracle MSE, learned MSE,
  and subset regret).
- Added cost-aware `subset-soft` router supervision. It trains against a soft
  distribution over all 70 E=8/K=4 subsets instead of only the single argmin
  subset.
- The benchmark now permits one-layer runs for causal controls.

## Correctness controls

On the Qwen3-0.6B E=8/K=4 smoke control with explicit scale 4:

| control | result |
|---|---:|
| token-loop hard training vs grouped inference, max absolute error | `1.02e-6` |
| token-loop hard training vs grouped inference, MSE | `4.34e-14` |
| full-active K=E parent reconstruction, max absolute error | `1.91e-6` |
| full-active K=E parent reconstruction, MSE | `1.23e-13` |

These pass the intended float32 numerical tolerance. The implementation
parity problem identified by the expert audit is therefore fixed for these
paths.

## Corrected two-layer reference

Protocol: Qwen/Qwen3-0.6B, layers 25–26, contiguous E=8/K=4 groups, explicit
hard scale 4, rank-64 cross-group correction, 300 child steps with the final
200 hard steps at LR 3e-4, 100 router-supervision steps, and the same held-out
development text.

| variant | seed 2026 | seed 2027 | decision |
|---|---:|---:|---|
| hard-label subset router | `+0.05445` | `+0.05297` | fail; not stable at the `+0.05` gate |
| hard block refinement, 100 steps, LR 1e-4 | `+0.05333` | — | fail; only `0.00112` better than baseline |
| subset-soft, temperature 0.25 | `+0.04874` | `+0.05079` | promising but not a two-seed pass |
| subset-soft, temperature 0.50 | — | `+0.05229` | reject for now |
| subset-soft, temperature 0.10 | — | `+0.05399` | reject for now |

The block objective did not repair the handoff by itself. The soft cost-aware
router is a real but small improvement; it must not yet be called the final
solution.

## Causal controls and oracle headroom

Single-layer corrected runs pass comfortably:

- layer 25 only: `+0.01422`;
- layer 26 only: `+0.03542`.

In the two-layer hard-label run, restoring one parent without retraining gives
`+0.03562` when layer 25 is restored and `+0.01422` when layer 26 is restored.
Each child therefore works alone; the combined failure is a cascade/interface
effect rather than a single unusable child.

The strongest result is the paired oracle control: the *same trained child
weights* are evaluated once with learned routing and once with exact
best-subset routing.

| seed | learned routing | paired exact oracle routing |
|---|---:|---:|
| 2026 | `+0.05445` | `+0.03282` |
| 2027 | `+0.05297` | `+0.03004` |

This is the clearest positive signal so far. The sparse decomposition contains
useful subsets; the deployable learned router is leaving quality on the table.

For seed 2026 hard-label routing, held-out exact-subset match was 38.99% on
layer 25 and 54.15% on layer 26. The corresponding reconstruction subset
regrets were 0.1702 and 0.1385 MSE. These diagnostics explain why the oracle
passes while the learned route misses the gate.

## Decision and next step

The fundamental operator mismatch is fixed, and the architecture is not
discarded: exact routing proves headroom. However, the learned router is not
yet stable enough for 4-layer scaling, 700M/1B scaling, or a deployment claim.

The next experiment should use a larger, non-repeated calibration corpus and
cost-aware router training with held-out route regret. Only after a corrected
two-seed 2-layer reference passes should we repeat the 4-layer reference. If
the oracle remains good but learned regret remains high, focus on router
generalization and multi-subset supervision; do not add another correction
cell or increase model capacity first.

The JSON artifacts for the runs above are kept under `results/runs/` locally;
that directory remains ignored by the repository, while this report records
the reproducible commands' effective settings and decisions.
