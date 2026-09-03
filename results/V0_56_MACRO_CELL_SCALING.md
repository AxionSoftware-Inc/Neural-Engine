# V0.56 Macro-Cell capacity scaling

Status: **positive architecture signal, but not yet a monotonic scaling law**

## Objective

The earlier 300M/500M/700M screens showed that increasing the virtual circuit
bank did not reliably improve quality. This experiment tests a different kind
of capacity: a reusable macro-cell is a short serial program of low-rank local
transforms, selected by a separate hierarchical top-1 router. The existing
attention-free Dynamic Register and its micro-circuit router remain unchanged.

The macro bank is optional and has no attention or global all-bank score. The
stored bank grows with `macro_cell_count`, while one macro-cell is active per
executed operation. The implementation is in `neural_engine/macro_cells.py`
and is integrated through `DynamicRegisterNeuralEngine`.

## Protocol

All runs use the same 20M virtual-tier Dynamic Register configuration, seed 17
unless stated otherwise, batch size 512, 3,000 optimizer steps, training on
depths 1–4, and evaluation on unseen depths 5–6. The macro-cell has rank 8,
local depth 4, candidate pool 4, and active top-1 routing. The benchmark has
no Transformer or attention path.

## Converged capacity screen

| Macro cells | Physical params | Macro active estimate | Train | Held-out 5–6 | Time | Macro utilization |
|---:|---:|---:|---:|---:|---:|---:|
| 16, seed 17 | 2.17M | 30.7K | 84.30% | 73.24% | 422s | not audited in run |
| 64, seed 17 | 3.44M | 32.3K | 82.96% | 71.48% | 434s | 100.00% |
| 256, seed 17 | 8.53M | 33.8K | 87.94% | **78.56%** | 450s | 98.83% |
| 256, seed 18 | 8.53M | 33.8K | 84.84% | **75.29%** | 436s | 97.66% |

The 256-cell two-seed mean is **76.93%**. The previous learned-only 20M
reference was 72.85%/75.49% across seeds 17/18 (mean 74.17%), so the new
macro path improves the two-seed mean by **2.76 percentage points**. The
seed-17 gain over its direct reference is 5.71 points; seed 18 is essentially
neutral. This is encouraging but not yet a reliable universal gain.

The 64-cell result is below the 16-cell result, so the curve is not monotonic.
This means that simply adding more independently learned macro rows is not a
complete capacity solution. The 256-cell route is nevertheless not dead: most
of the bank is used, the top macro receives only 2.69% of traffic in seed 17
and 3.73% in seed 18, and active macro parameters remain near 34K while stored
macro parameters grow to 6.79M.

## Short scaling screen

At 1,000 steps, the held-out scores were 11.43%, 12.01%, and 12.11% for 16,
64, and 256 cells. This gives a small early-training trend, but the converged
screen is the primary reference because the 64-cell run later regresses.

## Interpretation

This is the first positive result for the macro-cell direction. It suggests
that capacity can be added as reusable local programs rather than only as a
larger flat circuit bank. It does **not** yet justify a 1B/1T jump: the result
is non-monotonic, the second seed is weaker, and the task is still arithmetic
composition. The near-full route utilization also rules out simple dead-cell
collapse as the explanation for the 64-cell regression.

## Next experiment

The next step is staged macro growth, not a larger scratch bank:

1. train a 16-cell parent;
2. expand to 64/256 by cloning and perturbing trained macro rows;
3. preserve the learned router levels and warm up new routing capacity;
4. compare against scratch 256 with the same seed pair;
5. repeat on a longer-depth and non-arithmetic compositional task.

Only if parent-grown macro capacity beats scratch capacity on both seeds and
the harder task will the architecture be considered ready for a 300M+ screen.

## Reproduction run IDs

```text
ne_dynamic_20m_macro_v0_scale16_1000
ne_dynamic_20m_macro_v0_scale64_1000
ne_dynamic_20m_macro_v0_scale256_1000
ne_dynamic_20m_macro_v0_depth_holdout_3000
ne_dynamic_20m_macro_v0_64_depth_holdout_3000
ne_dynamic_20m_macro_v0_256_depth_holdout_3000
ne_dynamic_20m_macro_v0_256_depth_holdout_seed18_3000
```
