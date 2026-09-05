# V0.115 `taklif22` Operation-Swap Intervention

## Question

Can the machine adapt a changed computational rule while keeping the external
fact memory fixed?

## Protocol

Start from the structured scalar-lane machine with four latent cells, external
32-entry facts, 8 working-memory slots, and strong synthetic role/state
alignment. The original operation-0 rule is `x -> x + 0.10`. The intervention
changes only that rule to `x -> x - 0.15`; the fact table and fixed 2--4-step
cases remain unchanged. Before adaptation all parameters are frozen. Then only
cell 0 is trained for 1,000 steps on the changed rule.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --steps 4000 --structured-value-lane --state-supervision-weight 1.0 --role-loss-weight 1.0 --operation-adapt-steps 1000 --operation-adapt-learning-rate 0.002 --log-every 1000 --output results/runs/latent_computational_machine_v1_operation_swap_seed2026.json
```

## Results

| stage | operation-swap MSE | MAE | unchanged fact memory |
|---|---:|---:|:---:|
| before adapting cell 0 | 0.05456 | 0.1489 | yes |
| after adapting cell 0 | **0.00204** | **0.0205** | yes |

The route distribution stayed unchanged, with the same roughly balanced cell
usage as the original machine. Only cell 0 was trainable during adaptation;
the external fact table was never retrained or rewritten.

## Decision

**Modularity gate passes conditionally.** The machine can keep factual memory
fixed while changing one reusable computational skill, which is the intended
`taklif22` operation-swap behavior. This is stronger evidence than an accuracy
number alone because it is an intervention on the decomposition.

This does not erase V0.114: the same machine still fails 5--6-step depth OOD.
The next required control is a compact dense recurrent student with the same
state/active-compute budget. If it also fails depth OOD, the task or training
curriculum is the bottleneck; if it passes, the sparse cell dynamics remain the
problem.

## Artifact

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_operation_swap_seed2026_final.json`
