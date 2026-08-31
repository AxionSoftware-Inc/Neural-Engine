# Controlled counterfactual route sensitivity

The aggregate route-overlap result was followed by a controlled test. Each
pair starts from the same generated example and changes exactly one thing:

1. one operand is incremented modulo 64 while task and all other operands stay
   fixed;
2. the operation token is replaced by another task with the same arity and
   reasoning depth while all operand tokens stay fixed.

For each pair, the route set is the union of selected circuit IDs across the
executed steps. The main metric is Jaccard overlap between the base and
counterfactual route sets.

## Reproducibility

```powershell
python analyze_counterfactual_routes.py --checkpoint results/checkpoints/ne20_v09_full.pt --device cuda --examples-per-task 128 --output results/runs/ne20_v09_counterfactual_routes_128.json
python analyze_counterfactual_routes.py --checkpoint results/checkpoints/ne20_v12_full.pt --device cuda --examples-per-task 128 --output results/runs/ne20_v12_counterfactual_routes_128.json
```

## Summary

| Counterfactual | NE-V0.9 fixed | NE-V0.12 adaptive |
|---|---:|---:|
| Single operand: mean route Jaccard | 19.06% | 19.32% |
| Single operand: mean route change | 80.94% | 80.68% |
| Single operand: pairs with any step changed | 100.0% | 96.9% |
| Single task token: mean route Jaccard | 2.89% | **2.07%** |
| Single task token: mean route change | 97.11% | **97.93%** |
| Single task token: pairs with any step changed | 100.0% | 100.0% |

The operation-token experiment excludes `compose_add_mul` because this task has
no other task with the same arity and depth. Matching arity/depth keeps the
comparison from conflating a changed operation token with a changed sequence
shape or required reasoning depth.

## Interpretation

Both variants are strongly input-conditioned. A single operand change causes a
route change in essentially every pair and leaves only about 19% route overlap.
Changing only the operation token produces an even stronger route change, with
roughly 2% overlap. This supports the intended design: routing is not a fixed
global circuit subset and is not simply selecting one static expert per task.

V0.12's lower overlap for operation-token changes is consistent with its better
task-conditioned separation. Its lower “fraction of steps changed” should not
be read as weaker sensitivity: adaptive inference intentionally skips later
steps for many pairs, so skipped-vs-skipped comparisons reduce that metric. The
route-set Jaccard and “any step changed” metrics are the safer comparisons.

This is still a routing diagnostic, not a causal proof that every selected
circuit is semantically necessary. The next experiment should replay the base
route on the counterfactual input (and vice versa), then measure the resulting
accuracy/logit degradation. That will test whether route changes are functionally
used rather than merely correlated with the input.

