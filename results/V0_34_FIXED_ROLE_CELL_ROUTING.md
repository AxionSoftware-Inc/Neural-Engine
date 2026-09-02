# V0.34 fixed role-cell routing

Status: **negative scaling result; do not scale this version**

## Change

The learned role-anchor from V0.33 was replaced with nine deterministic,
interleaved role cells. Each cell received the full 32-candidate value router
pool, so the fallback split from V0.32 was removed.

## Full-domain results

| Model | All nine pairs | Two hidden pairs |
|---|---:|---:|
| 20M fixed role-cell | 57.93% | 78.28% |
| 100M fixed role-cell | 57.08% | not run |
| Existing references | 58.10% / 59.23% | 64.89% / 89.16% |

The 20M hidden result improved over the reference, but the all-pairs control was
slightly lower. At 100M the all-pairs score fell well below the reference, so
the hidden-stage run was not started.

## Decision

The fixed cells prevented learned anchor collapse but did not solve large-bank
underfitting. The larger cell-local search space still did not convert stored
rows into useful computation. This route topology is rejected for scaling.

## Next direction

The next experiment changes the circuit bank itself: a shared residual/basis
path will provide common computation to every route, while selected rows keep
small specialized adapters. This tests whether independent circuit rows are
the deeper credit-assignment bottleneck.
