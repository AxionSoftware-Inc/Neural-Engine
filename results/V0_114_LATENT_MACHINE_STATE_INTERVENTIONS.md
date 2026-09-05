# V0.114 `taklif22` State and Credit Interventions

## Question

Is the V0.113 long-depth failure repaired by the state equation, a compact
value register, intermediate-state supervision, or bounded writes?

## Protocol

All variants use the same 92K-class latent machine task unless noted: external
32-entry facts, four latent cells, 8 working-memory slots, 2--4-step training,
paired fixed cases, and 5--6-step evaluation. The role-loss and
intermediate-state targets are synthetic diagnostic aids, not required labels
for deployment.

## Results

| variant | in-range MSE | OOD 5--6-step MSE | route/control observation |
|---|---:|---:|---|
| dense state baseline | 0.0098 | 62.83 | balanced routes, strong drift |
| dense state, train 2--6 | 0.0157 | 51.17 | curriculum helps slightly, gate still fails |
| dense state, scale 0.25 + LayerNorm | 0.0099 | 65.00 | prescribed stabilization did not help |
| scalar lane, role 1 + state target 1 | 0.0025 | 53.11 | best OOD reduction, still fails |
| scalar lane, role 0.05 + state target 1 | 6.24 | 89.30 | route alignment collapses |
| scalar lane + clip 1.5 | 0.0894 | 67.48 | bounded write harms fit |

The scalar-lane best variant uses only 97 parameters per active computation
cell, but still fails the 5--6-step gate. The 2--6 training curriculum lowers
OOD MSE from 62.83 to 51.17 without changing the graph, so training exposure
is part of the problem, but it does not explain the failure completely.

## Decision

**No state intervention passes the variable-depth quality gate.** The results
support three conclusions:

1. route usage is not the bottleneck — cells remain broadly used;
2. fact memory is genuinely consumed — zeroing it causes a large degradation;
3. the current learned cell dynamics do not extrapolate stably across repeated
   composition, even with explicit state targets or bounded writes.

The current V1 cell/state implementation is therefore rejected as a validated
`taklif22` architecture. Do not scale its cells or transfer it to Qwen yet.
The next diagnostic is the required operation-swap intervention with the fact
table held fixed, followed by a compact dense recurrent control. If the dense
control also fails the same depth gate, the task's extrapolation protocol is
too demanding for this training budget; if it passes, the sparse cell dynamics
are specifically at fault.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_fixed_seed2026.json`
- `results/runs/latent_computational_machine_v1_train2to6_seed2026.json`
- `results/runs/latent_computational_machine_v1_stabilized_seed2026.json`
- `results/runs/latent_computational_machine_v1_state_role1_seed2026.json`
- `results/runs/latent_computational_machine_v1_state_supervised_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_seed2026.json`
