# V0.58 Long-Depth Macro-Cell Gate

## Purpose

The standard dynamic-composition gate evaluates unseen depths 5--6. This can
hide errors that accumulate over longer recurrent execution. V0.58 keeps the
same attention-free 256-cell Macro-Cell architecture, trains on depths 1--4,
and evaluates on unseen depths 5--8.

## Results

Each run uses 3,000 optimizer steps and 1,024 evaluation examples per depth.

| Seed | Train accuracy | Depth 5 | Depth 6 | Depth 7 | Depth 8 | Mean eval |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 87.74% | 82.03% | 80.18% | 74.41% | 75.39% | 78.00% |
| 18 | 82.59% | 71.39% | 70.51% | 66.70% | 66.80% | 68.85% |

The two-seed mean is **73.43%**. Macro routing itself remains broad: the
evaluation audit reaches 252/256 cells for seed 17 and 256/256 for seed 18.
Therefore this result does not look like a dead-cell or top-route collapse.
Quality declines as execution depth increases, and the train accuracy at
depths 3--4 is still low after 3,000 steps. The gate is consequently a useful
failure signal, but not yet a fully converged capacity verdict.

## Interpretation

The current evidence points to a recurrent-state optimization/generalization
bottleneck: the model can route many cells, but errors accumulate across more
than six serial updates. A longer training budget is required before changing
the architecture. If 9,000-step training still shows the same depth slope,
the next architectural work should target state normalization and a stable
per-step update interface, not a larger Macro-Cell bank.

The 3,000-step checkpoints are recorded in:

- `results/runs/ne_dynamic_20m_macro_v0_256_long_depth_seed17_3000.json`
- `results/runs/ne_dynamic_20m_macro_v0_256_long_depth_seed18_3000.json`

