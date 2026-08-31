# Route swap ablation

The replay experiment showed that a mismatched route can damage a
counterfactual computation. This follow-up measures the dose-response: replace
the route for 0%, 25%, 50%, or 100% of a balanced batch, either with a route
from any other task (`global`) or with a route from another example of the same
task (`within_task`).

The models use fixed three-step execution in this diagnostic so route tensors
have the same shape. V0.12's learned halt head is therefore disabled only for
this ablation; its weights and ordinary adaptive behavior are unchanged.

## Reproducibility

```powershell
python analyze_route_ablation.py --checkpoint results/checkpoints/ne20_v09_full.pt --device cuda --examples-per-task 64 --output results/runs/ne20_v09_route_ablation_64.json
python analyze_route_ablation.py --checkpoint results/checkpoints/ne20_v12_full.pt --device cuda --examples-per-task 64 --output results/runs/ne20_v12_route_ablation_64.json
```

Each batch contains 64 examples for each of the 15 task IDs (960 examples).
The swapped route includes the source example's circuit IDs, circuit weights,
and route gains; the current input's encoder/state/output remain in place.

## Global route swapping

| Swapped route fraction | V0.9 accuracy | V0.9 loss increase | V0.12 accuracy | V0.12 loss increase |
|---:|---:|---:|---:|---:|
| 0% | 71.98% | +0.0000 | 71.77% | +0.0000 |
| 25% | 71.77% | +0.0145 | 71.56% | +0.0098 |
| 50% | 70.94% | +0.0267 | 71.15% | +0.0285 |
| 100% | 70.00% | +0.0536 | 70.42% | +0.0535 |

At 100% global swapping, accuracy falls by 1.98 percentage points for V0.9
and 1.35 points for V0.12. Loss rises monotonically with the amount of route
replacement in both models.

## Within-task route swapping

| Swapped route fraction | V0.9 accuracy | V0.9 loss increase | V0.12 accuracy | V0.12 loss increase |
|---:|---:|---:|---:|---:|
| 0% | 71.98% | +0.0000 | 71.77% | +0.0000 |
| 25% | 71.15% | +0.0166 | 71.25% | +0.0070 |
| 50% | 70.52% | +0.0338 | 71.04% | +0.0198 |
| 100% | 69.69% | +0.0556 | 70.42% | +0.0368 |

Even routes exchanged between examples of the same task cause a measurable
quality loss. This is consistent with operand-conditioned computation rather
than only task-level routing.

## Interpretation

The 0% rows are an identity control: forcing each example to use its own
recorded route reproduces the natural result. The monotonic degradation as
routes are replaced provides a stronger causal signal than route Jaccard alone.

The effect remains modest, so the shared encoder, recurrent controller, and
output head still carry substantial task information. This is desirable for
robustness but means route quality is not the sole source of the current
accuracy. As in the replay report, the diagnostic still evaluates the router
internally while discarding its selected IDs; it measures dependence on the
selected circuit path, not a standalone router-speed benchmark.

**Decision:** keep dynamic route selection. The next research comparison is a
controlled active-circuit budget sweep (4, 8, and 16 selected circuits) on the
same checkpoint, followed by a second training seed to check whether the
causal route signal is stable.

