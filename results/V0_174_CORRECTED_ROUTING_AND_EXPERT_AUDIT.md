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
| subset-soft, router hidden 64 | — | `+0.05490` | reject; width alone is not the fix |
| subset-soft, router hidden 256 | — | `+0.08855` | reject; unstable child-2 training |
| independent energy router | `+0.08221` | — | reject |

The block objective did not repair the handoff by itself. The soft cost-aware
router is a real but small improvement; it must not yet be called the final
solution.

Two additional controls reject simple data/width explanations. Using 12
calibration batches from the broader repository README instead of the matched
Qwen calibration text worsened seed 2026 to `+0.08689` (paired oracle
`+0.04375`), so more distribution-shifted text is not automatically useful.
Reducing the router hidden width from 128 to 64 worsened seed 2027 to
`+0.05490`; increasing it to 256 made child-2 local MSE `4.57` and final
delta `+0.08855`. Router width is therefore not the current solution.

## Active-budget control: E=8/K=6

After the K=4 router controls, the same corrected two-layer protocol was run
with six of eight groups active and explicit hard scale 6. This is a lower
sparsity setting (75% active), so it tests whether the earlier quality loss was
primarily caused by an overly aggressive active budget rather than by the
operator itself.

| seed | learned CE delta | paired exact-oracle CE delta | active fraction | sparse-bank timing / parent |
|---|---:|---:|---:|---:|
| 2026 | `+0.03619` | `+0.02186` | `75%` | `1.281x` |
| 2027 | `+0.03837` | `+0.02574` | `75%` | `1.282x` |

Both seeds pass the corrected `+0.05` quality gate, making K=6 the first
stable learned-router result after the parity fixes. The oracle is also good,
but the gap between learned and oracle routing remains. The result is not yet
a deployment win: the current grouped sparse-bank implementation is about
28% slower end-to-end than the dense parent, and only 25% of the expert groups
are removed. This is evidence that the capacity problem is partly active-budget
related, not proof that arbitrary scale-up will continue improving quality.

## Four-layer K=6 depth control

The two-layer K=6 setting was then extended to four consecutive layers with
the same training and routing protocol.

| seed | learned CE delta | paired exact-oracle CE delta | quality gate | timing / parent |
|---|---:|---:|---|---:|
| 2026 | `+0.05391` | `+0.04257` | learned fail, oracle pass | `1.566x` |
| 2027 | `+0.01794` | `+0.00801` | pass | `1.567x` |

This is a mixed result, not a clean four-layer pass. The oracle passes on both
seeds, so the sparse cells still compose usefully at this depth. The main
seed-dependent failure is child optimization: the layer-26 local evaluation
MSE is `3.20` for seed 2026 versus `0.46` for seed 2027. Routing regret is
present, but it is not sufficient to explain that large difference. The next
control therefore reduces the hard-phase learning rate before changing the
architecture or increasing model size.

## Training-operator parity controls

The mixed result above came from 300 soft-calibration steps followed by 200
hard-route steps. That handoff trains one operator and then abruptly optimizes
another. The following seed-2026 controls isolate that transition:

| training protocol | learned CE delta | paired oracle CE delta | decision |
|---|---:|---:|---|
| handoff, hard LR `3e-4` | `+0.05391` | `+0.04257` | borderline/fail |
| handoff, hard LR `1e-4` | `+0.31384` | `+0.27967` | reject; child 3 diverged |
| no hard phase (`0/300`) | `+1.53048` | `+1.52453` | reject; soft operator is not a sparse endpoint |
| soft-to-hard blend, 100 steps | `+0.08783` | `+0.10253` | reject; blended operator worsened the cascade |
| direct hard (`300/300`) | `+0.01731` | `+0.00486` | pass |

Direct-hard was then repeated on seed 2027:

| seed | learned CE delta | paired exact-oracle CE delta | child local MSE range | timing / parent |
|---|---:|---:|---:|---:|
| 2026 | `+0.01731` | `+0.00486` | `0.300–0.339` | `1.566x` |
| 2027 | `+0.01535` | `+0.00378` | `0.310–0.342` | `1.562x` |

This is the first stable four-layer K=6 result across both seeds. It changes
the interpretation of the earlier failure: the transferred sparse cells and
router are viable, but a soft-to-hard training mismatch destabilizes the
composition. The direct-hard recipe is a training control, not yet a runtime
win; grouped sparse execution remains about 1.56x slower than the dense
parent.

## Eight-layer K=6 depth control

The direct-hard recipe was extended to the full adjacent Qwen3-0.6B depth
control, layers 19–26, with six of eight groups active at every layer.

| seed | learned CE delta | paired exact-oracle CE delta | child local MSE range | timing / parent |
|---|---:|---:|---:|---:|
| 2026 | `+0.01103` | `-0.00649` | `0.181–0.307` | `2.082x` |
| 2027 | `+0.02117` | `-0.00088` | `0.179–0.317` | `2.129x` |

Both seeds pass the `+0.05` gate. This is the strongest depth result so far:
the attention-free sparse FFN replacements compose across eight consecutive
layers, and paired oracle routing is essentially at parent quality. The
remaining limitation is efficiency: K=6 leaves 75% of groups active, yet the
current Python/PyTorch grouped bank is about 2.1x slower end-to-end because
dispatch and bank overhead dominate the saved expert work.

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

Replacing the 70-class subset predictor with eight independent energy scores
and top-4 selection worsened the corrected seed-2026 run to `+0.08221`. The
problem is therefore not solved by simply changing the router output format.

For seed 2026 hard-label routing, held-out exact-subset match was 38.99% on
layer 25 and 54.15% on layer 26. The corresponding reconstruction subset
regrets were 0.1702 and 0.1385 MSE. These diagnostics explain why the oracle
passes while the learned route misses the gate.

## Decision and next step

The fundamental operator mismatch is fixed, and the architecture is not
discarded: exact routing proves headroom. K=6 now has a stable two-seed
quality result, but K=4 remains unstable and the runtime is still worse than
the dense parent. Therefore this is not yet a general scaling law or a
deployment claim for 700M/1B.

The direct-hard eight-layer K=6 reference is now stable across two seeds,
while the runtime is still worse than the dense parent. The next quality
experiment is eight-layer direct-hard K=4, which tests the intended 50%-active
operating point under the corrected training operator. Runtime optimization is
a separate workstream; do not interpret the current 2.1x timing as a
deployment result.

The JSON artifacts for the runs above are kept under `results/runs/` locally;
that directory remains ignored by the repository, while this report records
the reproducible commands' effective settings and decisions.
