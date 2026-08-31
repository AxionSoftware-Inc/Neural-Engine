# Route stability and task overlap

This experiment checks whether the circuit router collapses onto the same
circuits for every input. It uses the trained full-distribution checkpoints,
128 fresh examples per task, and the same evaluation split for both models.
For each example, the active circuit set is the union of selected circuits
across its executed recurrent steps.

## Reproducibility

```powershell
python analyze_routes.py --checkpoint results/checkpoints/ne20_v09_full.pt --device cuda --examples-per-task 128 --output results/runs/ne20_v09_routes_128.json
python analyze_routes.py --checkpoint results/checkpoints/ne20_v12_full.pt --device cuda --examples-per-task 128 --output results/runs/ne20_v12_routes_128.json
```

The JSON files are ignored run artifacts. The checkpoint binaries remain local
as described in `results/CHECKPOINTED_INFERENCE.md`.

## Summary

| Metric | NE-V0.9 fixed | NE-V0.12 adaptive |
|---|---:|---:|
| Circuit bank | 1,408 | 1,408 |
| Circuits used | 1,379 (97.9%) | 1,357 (96.4%) |
| Dead circuit fraction | 2.06% | 3.62% |
| Always-hot circuits (>=50% examples) | 0.00% | 0.00% |
| Routing entropy | 6.475 nats | 6.453 nats |
| Maximum circuit load | 1.44% | 2.00% |
| Mean within-task sample Jaccard | 8.78% | 7.87% |
| Mean task-union Jaccard | 31.22% | **19.43%** |
| Average executed steps | 3.00 | 1.60 |

The within-task Jaccard compares different operand instances of the same task.
The task-union Jaccard compares the union of routes observed for two different
task IDs. These two Jaccard measurements have different set sizes and should
not be compared directly; the useful comparison is how the same metric changes
from V0.9 to V0.12.

## Per-task route coverage

| Task | V0.9 union | V0.9 within | V0.12 union | V0.12 within |
|---|---:|---:|---:|---:|
| add | 584 | 5.23% | 388 | 2.36% |
| subtract | 617 | 4.90% | 418 | 2.04% |
| multiply | 674 | 6.09% | 336 | 6.20% |
| greater_than | 596 | 3.87% | 271 | 7.61% |
| less_equal | 528 | 5.39% | 174 | 11.46% |
| xor_parity | 294 | 13.37% | 90 | 19.86% |
| max3 | 763 | 3.39% | 355 | 3.81% |
| median3 | 721 | 3.91% | 386 | 4.23% |
| min3 | 743 | 3.51% | 475 | 1.45% |
| reverse_sum | 480 | 13.05% | 607 | 2.63% |
| lookup | 795 | 2.86% | 845 | 1.60% |
| chain3 | 247 | 26.07% | 273 | 12.85% |
| compose_add_mul | 544 | 9.29% | 488 | 22.46% |
| compose_if | 999 | 1.41% | 914 | 1.77% |
| state_machine | 240 | 29.34% | 308 | 17.79% |

## Interpretation

The router is healthy in both variants: it uses almost the whole bank, has no
always-hot circuit group, and does not collapse to one fixed route. V0.12
reduces task-union overlap from 31.22% to 19.43%, which is consistent with more
task-conditioned routing after numeric encoding and adaptive execution.

The low within-task overlap is also informative. Different operand values can
select different paths even when the task ID is unchanged, so the system is
not merely implementing one static expert per task. This is evidence of
input-conditioned routing, not yet proof that individual circuits have clean
semantic roles: values and task identity are entangled in the current input
representation.

## Decision and next test

Keep V0.12 as the main variant. The next route experiment should use controlled
counterfactual pairs: hold the task and most operands fixed, change exactly one
operand or operation token, and measure which circuit decisions change. That
will separate value sensitivity from task specialization more cleanly than
aggregate Jaccard alone.

