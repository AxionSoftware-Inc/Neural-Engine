# V0.58 Long-Depth Macro-Cell Gate

## Purpose

The standard dynamic-composition gate evaluates unseen depths 5--6. This can
hide errors that accumulate over longer recurrent execution. V0.58 keeps the
same attention-free 256-cell Macro-Cell architecture, trains on depths 1--4,
and evaluates on unseen depths 5--8.

## Results

Each run uses 1,024 evaluation examples per depth. The preliminary screen uses
3,000 optimizer steps; both seeds were then continued to 9,000 steps.

| Seed | Steps | Train accuracy | Depth 5 | Depth 6 | Depth 7 | Depth 8 | Mean eval |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 3,000 | 87.74% | 82.03% | 80.18% | 74.41% | 75.39% | 78.00% |
| 18 | 3,000 | 82.59% | 71.39% | 70.51% | 66.70% | 66.80% | 68.85% |

The preliminary two-seed mean is **73.43%**. Macro routing itself remains
broad: the evaluation audit reaches 252/256 cells for seed 17 and 256/256 for
seed 18. Quality declines as execution depth increases, and train accuracy at
depths 3--4 is still low after 3,000 steps.

Both runs were then continued to 9,000 steps:

| Seed | Steps | Train accuracy | Depth 5 | Depth 6 | Depth 7 | Depth 8 | Mean eval |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 9,000 | 99.93% | 99.61% | 99.51% | 99.41% | 99.22% | 99.44% |
| 18 | 9,000 | 99.88% | 99.41% | 99.90% | 99.02% | 99.41% | 99.44% |

The converged two-seed mean is **99.44%**, with 254--256/256 cells reachable.
The earlier failure was therefore an optimization-budget issue, not a
dead-cell or top-route collapse.

## Interpretation

The current evidence shows that the recurrent state and Macro-Cell routing can
generalize through eight serial operations once the model is converged. The
large 3,000-to-9,000-step improvement means the training budget must scale with
execution depth. No state-normalization or Macro-Cell architecture change is
justified by this gate yet.

The 3,000-step checkpoints are recorded in:

- `results/runs/ne_dynamic_20m_macro_v0_256_long_depth_seed17_3000.json`
- `results/runs/ne_dynamic_20m_macro_v0_256_long_depth_seed18_3000.json`
- `results/runs/ne_dynamic_20m_macro_v0_256_long_depth_seed17_9000.json`
- `results/runs/ne_dynamic_20m_macro_v0_256_long_depth_seed18_9000.json`
