# Route replay causality test

Route sensitivity shows that the model changes circuit paths when its input
changes. This experiment asks the stronger question: does forcing the old path
actually damage the new computation?

For every counterfactual pair, the model first records the base route. The
counterfactual is then evaluated normally and once more with the base circuit
IDs, selected circuit weights, and route gains replayed. The reverse direction
is measured as well. Input encoding, recurrent state, and output head remain
those of the current input; only the recorded circuit path is swapped.

To align route tensors, both models use fixed three-step execution in this
diagnostic (`adaptive=False`). This does not change their weights or training
checkpoints; it only removes early-exit differences from the causal comparison.

## Reproducibility

```powershell
python analyze_route_replay.py --checkpoint results/checkpoints/ne20_v09_full.pt --device cuda --examples-per-task 128 --output results/runs/ne20_v09_route_replay_128.json
python analyze_route_replay.py --checkpoint results/checkpoints/ne20_v12_full.pt --device cuda --examples-per-task 128 --output results/runs/ne20_v12_route_replay_128.json
```

## Results

### Single operand changed

| Metric | NE-V0.9 fixed | NE-V0.12 adaptive |
|---|---:|---:|
| Counterfactual natural accuracy | 71.93% | 72.19% |
| Counterfactual with base route | 70.36% | 71.04% |
| Accuracy change | -1.56 pp | -1.15 pp |
| Counterfactual loss increase | +0.0489 | +0.0332 |
| Reverse-direction base loss increase | +0.0629 | +0.0423 |

### Single operation token changed

The replacement operation has the same arity and reasoning depth as the base
operation, so sequence shape and expected step count remain comparable.

| Metric | NE-V0.9 fixed | NE-V0.12 adaptive |
|---|---:|---:|
| Counterfactual natural accuracy | 79.45% | 79.93% |
| Counterfactual with base route | 77.94% | 78.91% |
| Accuracy change | -1.50 pp | -1.02 pp |
| Counterfactual loss increase | +0.0496 | +0.0278 |
| Reverse-direction base loss increase | +0.0491 | +0.0235 |

## Interpretation

Replaying a mismatched route consistently increases loss and reduces exact
accuracy in both directions. This is a positive causal signal: the route
changes measured earlier are functionally involved in the computation rather
than being a logging artifact or an unused side signal.

The effect is modest because the circuit bank is only one part of the model;
the shared encoder, recurrent controller, and output head can still recover
some of the answer. The replay implementation also still evaluates the router
for diagnostic statistics, although its selected IDs are discarded, so this is
a circuit-path causality test rather than a wall-clock routing-ablation test.

**Decision:** retain dynamic routing as a real architectural mechanism. The
next useful ablation is route randomization or cross-task route swapping at
several percentages, which can estimate how much of the final quality depends
on the selected circuits versus shared computation.

