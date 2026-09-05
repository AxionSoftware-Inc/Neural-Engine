# V0.117 `taklif22` Absolute-State Transition

## Question

Does asking each selected cell to predict the next scalar-register value,
rather than an additive delta, prevent accumulated state drift?

## Protocol

The V0.114 structured scalar-lane machine is held fixed: four latent cells,
top-1 route, 8 working-memory slots, external facts, role-loss weight 1.0,
intermediate-state supervision weight 1.0, and 2--4-step training. In the
absolute variant, the selected cell output is used as the next register value
instead of being added as a delta. No Qwen or operator-valued parameters are
used.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --steps 4000 --structured-value-lane --state-supervision-weight 1.0 --role-loss-weight 1.0 --transition-mode absolute --log-every 1000 --output results/runs/latent_computational_machine_v1_absolute_seed2026.json
```

## Results

The model keeps broad route usage (about 24--26% per cell), but aggregate
in-range quality regresses to MSE 0.302 and 5--6-step OOD MSE is 56.21. The
fixed-depth curve localizes the failure:

| exact depth | MSE |
|---:|---:|
| 2 | 0.0022 |
| 3 | 0.0071 |
| 4 | 0.0412 |
| 5 | 0.0613 |
| 6 | **8.2828** |

The absolute transition improves the depth-5 point relative to the unstable
delta path, but it still fails at depth 6 and loses the short-program accuracy
needed for a useful machine. Fact table B in-range MSE is 0.2762, and zeroing
memory still degrades to 67.24.

## Decision

**Absolute transition is rejected as the canonical state update.** It confirms
that bounded/absolute state proposals can delay error growth by one depth, but
they do not provide stable compositional extrapolation. The fact and memory
interventions remain meaningful, while the depth failure persists across both
routed and compact dense controls.

The next `taklif22` change should be a mathematically stable/equivariant cell
transition or a revised task with a clearly bounded state domain. Do not add
more cells, scale the memory, or return to Qwen transfer until a depth curve
passes this control.

## Artifact

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_absolute_seed2026.json`
