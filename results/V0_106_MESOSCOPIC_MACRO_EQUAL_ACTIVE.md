# V0.106 Mesoscopic MacroCell Equal-Active-Budget Gate

## Question

At approximately the same active body budget as the current micro-circuit
reference, can two richer mesoscopic MacroCells replace many small routed
transforms on unseen program depths?

## Canonical configuration

This is the isolated `taklif20` V1 body: 64 independent cells, state width
384, hidden width 480, bilinear rank 128, candidate pool 8, and active top-2
serial execution. Each cell has exactly 897,504 scalar parameters, so the two
active cells expose 1,795,008 active body parameters. There is no factorized
MacroCell bank, operator-valued parameterization, attention, or inner router.
The test trains on depths 1--4 and evaluates on unseen depths 5--8.

The first smoke run with residual scale 1.0 produced non-finite held-out
values. The canonical run therefore uses the smallest stabilizing residual
scale tested, 0.1, and is recorded separately from that failed smoke.

```powershell
python -u experiment_mesoscopic_macro.py --device cuda --steps 3000 --batch-size 64 --examples-per-depth 128 --log-every 500 --residual-scale 0.1 --seed 17 --run-id ne_mesoscopic_macro_64_equal_active_seed17_3000 --output results/runs/ne_mesoscopic_macro_64_equal_active_seed17_3000.json
python -u experiment_mesoscopic_macro.py --device cuda --steps 3000 --batch-size 64 --examples-per-depth 128 --log-every 500 --residual-scale 0.1 --seed 18 --run-id ne_mesoscopic_macro_64_equal_active_seed18_3000 --output results/runs/ne_mesoscopic_macro_64_equal_active_seed18_3000.json
```

## Results

| seed | train accuracy | held-out accuracy | held-out depths | train loss | held-out loss |
|---:|---:|---:|---|---:|---:|
| 17 | 6.64% | 4.88% | 4.69%, 3.91%, 4.69%, 6.25% | 3.64 | NaN |
| 18 | 7.62% | 3.71% | 1.56%, 3.13%, 3.13%, 7.03% | 3.59 | NaN |
| **mean** | **7.13%** | **4.30%** | — | — | — |

The held-out loss is NaN because the long-depth states/logits become
non-finite; the displayed argmax accuracy is therefore not a trustworthy
quality measure and is reported only to expose the failure. The route audit
does not show dead routing: 56--64 of the 64 cells are reached. The failure is
optimization/state stability under hard top-2 routing, not simple bank
under-utilization.

For context, the earlier V0.58/V0.59 MacroCell results are not a positive
counterexample to this gate: those models augmented the existing micro-circuit
engine and used a different small rank-8 cell body, rather than replacing the
micro path with the canonical equal-active-budget cell.

## Decision

Reject the canonical mesoscopic MacroCell replacement at this gate. Do not
increase cell count, hidden width, or stored MacroCell capacity. A future
MacroCell attempt would need an independently justified training/credit
assignment mechanism and a finite-state stability test before any larger run.
Continue to the independent self-describing semantic-routing proposal.

## Artifacts

- `neural_engine/mesoscopic_macro.py`
- `experiment_mesoscopic_macro.py`
- `tests/test_mesoscopic_macro.py`
- `results/runs/ne_mesoscopic_macro_64_equal_active_seed17_3000.json`
- `results/runs/ne_mesoscopic_macro_64_equal_active_seed18_3000.json`
