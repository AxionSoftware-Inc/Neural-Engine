# V0.116 `taklif22` Dense Recurrent Control

## Question

Is the 5--6-step failure specific to sparse routed cells, or does a compact
dense recurrent function with the same external-memory/state task fail too?

## Protocol

The control keeps the external fact table, 8 working-memory slots, scalar
register, operation embeddings, and 2--4-step training protocol. It removes
the latent cell bank and router: one shared 1,121-parameter dense operation
function receives the current scalar and operation embedding at every step.
The model has 34,689 stored parameters. Four fixed paired batches are used for
each evaluation intervention.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --model-kind dense-control --steps 4000 --role-loss-weight 0.0 --state-supervision-weight 1.0 --log-every 1000 --output results/runs/latent_computational_machine_v1_dense_control_seed2026_final.json
```

## Results

| evaluation | MSE | MAE | interpretation |
|---|---:|---:|---|
| in-range 2--4 steps | 0.01189 | 0.0224 | fits short programs |
| unseen 5--6 steps, fact table A | 63.61 | 0.242 | fails the same depth gate |
| same in-range cases, swapped fact table B | 0.00109 | 0.0194 | external facts generalize |
| zero-memory intervention | 67.60 | 0.425 | memory is materially used |

The routed V0.113 baseline has in-range MSE 0.0098 and OOD MSE 62.83. The
dense control is therefore similarly good in-range and similarly poor at
depth, despite removing routing entirely.

## Decision

**The depth failure is not sufficient evidence of a sparse-router failure.** A
compact dense recurrent control exhibits the same pattern: good short-program
fit, fact-swap preservation, strong zero-memory degradation, and catastrophic
5--6-step extrapolation. The current synthetic depth protocol is primarily a
training/stability challenge for the learned scalar transformations.

This control does not validate the full `taklif22` thesis, and it does not
justify Qwen transfer. It does show that further router tuning alone is not the
right next move. The next architecture work should use a bounded or
equivariant state transition with an explicit stability test, and compare it
against this dense control before adding more cells.

## Artifact

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_dense_control_seed2026_final.json`
