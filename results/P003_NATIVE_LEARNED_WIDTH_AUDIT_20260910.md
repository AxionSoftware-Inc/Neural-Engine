# P-003 native learned dynamic-width audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Base: 500M stable-prefix K=16 checkpoint, seeds 17/18

## Question

Can a small predictor reproduce the useful part of the K=8/K=16 width oracle
without computing both routes first?

## Method

- The circuit bank, recurrent body, and base route keys were frozen.
- A single linear width head (`384 → 1`) was trained independently for each
  seed on paired K=8/K=16 step losses from a calibration stream.
- The target was `K=16` only when its step cross-entropy improvement exceeded
  `lambda=0.05 * (1.0 - 0.5)`. The head saw query states from both fixed-width
  trajectories so it was not trained only on one state distribution.
- At inference the head sees the current state before circuit execution and
  chooses K=8 or K=16. Only the chosen route is executed; the top-16 router
  candidates remain available for the wide branch.
- Evaluation used a separate OOD stream: 24 batches each for uniform,
  combination-heldout, low-edge, and high-edge conditions.

## Result: two-seed means

| Condition | K=16 exact | Learned exact | Delta | K=16 hard mean | Learned hard mean | Mean width |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 82.999% | **83.008%** | **+0.009 pp** | 58.583% | **58.594%** | 8.99 |
| Combination holdout | 83.095% | **83.099%** | **+0.004 pp** | 58.952% | **58.974%** | 8.98 |
| Low edge | 98.177% | 98.177% | 0.000 pp | 95.757% | 95.757% | 8.41 |
| High edge | 98.320% | **98.333%** | **+0.013 pp** | 95.877% | **95.909%** | 8.50 |

Uniform mean width was `8.99/16 = 56.2%` of the K=16 active circuit budget;
only `12.4%` of executed steps used the wide K=16 path. Combination holdout
was nearly identical at `8.98/16`; edge probes were even narrower at about
`8.4–8.5` circuits on average.

The learned checkpoints add only 385 parameters for the width head. The audit
counts actual executed IDs separately from the router’s full top-16 decision,
so the width reduction is not a bookkeeping artifact. A wall-clock kernel
benchmark is still required: two grouped dispatch calls can reduce the
theoretical slot cost without giving the same proportional latency on a small
GPU batch.

## Interpretation

This is the first deployable-path positive result for the active-width idea:
the predictor approaches the oracle’s roughly half-width operating point while
preserving the K=16 quality ceiling on this two-seed OOD screen. It also
explains why entropy-gating was weak: entropy was not sufficiently aligned with
the actual K=8 versus K=16 loss gap.

The result is not yet a default change. Calibration labels came from frozen
paired trajectories, and the predictor was not jointly trained with the
recurrent body. The next controls are a third seed, longer continuation, a
true wall-clock profile, and a regret audit that penalizes sending hard examples
to K=8 more than sending easy examples to K=16.

## Decision

`PROMISING OPT-IN — QUALITY/PARAMETER GATE PASSED; RUNTIME AND THIRD-SEED OPEN`.

Keep the learned-width checkpoints and head opt-in. Do not replace the K=8 or
K=16 defaults yet. If the third-seed and runtime controls hold, this becomes the
preferred route for larger native banks: capacity can remain high while active
parameters are selected per example.

## Raw evidence and reproduction

- [Learned-width OOD JSON](diagnostic_native_learned_width_20260910.json)
- [Head-training JSON](diagnostic_native_learned_width_training_20260910.json)
- [Head training benchmark](../benchmark_native_learned_width.py)
- [K=16 source configuration](../configs/ne_500m_v12_factorized_shared_routekeys_step_adapter_two_edge_mix_stable_prefix_active16.yaml)

```powershell
python benchmark_native_learned_width.py `
  --checkpoints results/checkpoints/ne500_stable_prefix_active16_s17_3000.pt `
    results/checkpoints/ne500_stable_prefix_active16_s18_3000.pt `
  --batches 8 --epochs 20 --lambda-target 0.05 `
  --output results/diagnostic_native_learned_width_training_20260910.json

python audit_native_ood.py `
  --checkpoints results/checkpoints/ne500_stable_prefix_active16_s17_3000_learned_width.pt `
    results/checkpoints/ne500_stable_prefix_active16_s18_3000_learned_width.pt `
  --batches 24 `
  --output results/diagnostic_native_learned_width_20260910.json
```
