# V0.118 `taklif22` Bounded Stable-Cell Control

## Question

Does the `taklif22` decomposition work when the state domain is bounded, and
can the latent route emerge without synthetic operation labels?

## Protocol

The synthetic task uses four bounded transformations that preserve a scalar
state range: affine update, scale, negate, and half-square. The machine has an
external 32-entry fact table, 8 working-memory slots, four computation cells,
and top-1 execution. Each cell uses the stable `1,x,x²` delta basis, which is a
diagnostic equivariant control for this task. Training uses 2--4 steps and
evaluation uses paired fixed cases at depths 2--6, fact-table swap, and
zero-memory intervention.

Two routing regimes are compared:

- role-aligned: synthetic operation-role cross-entropy is enabled, as allowed
  by `taklif22` Phase C for a diagnostic stage;
- labels-free: role loss is zero, so only task/state losses can specialize the
  latent route.

Role-aligned command:

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded --steps 4000 --structured-value-lane --structured-cell-kind polynomial --state-supervision-weight 1.0 --role-loss-weight 1.0 --operation-adapt-steps 1000 --operation-adapt-learning-rate 0.002 --log-every 1000 --output results/runs/latent_computational_machine_v1_bounded_polynomial_operation_swap_seed2026.json
```

Labels-free command:

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded --steps 4000 --structured-value-lane --structured-cell-kind polynomial --role-loss-weight 0.0 --state-supervision-weight 1.0 --log-every 1000 --output results/runs/latent_computational_machine_v1_bounded_polynomial_no_role_seed2026.json
```

## Results

### Role-aligned diagnostic

The role-aligned machine reaches exact-depth MSE from `1.08e-7` to
`2.60e-7` at depths 2--6. Fact-table B MSE is `1.82e-7`, zeroing memory
raises MSE to `0.0034`, and adapting only cell 0 to the changed operation
reaches `1.51e-6`. Cell usage remains broad at about 24--26% each.

### Labels-free control

Without role labels, the same task/state losses do not produce useful latent
specialization. In-range MSE is `0.3677`, depth-6 MSE is `0.4633`, and cell 0
receives only `0.7%` of routes while cells 1--3 absorb the rest. Same-program
route reuse falls to `0.889`, while different-program overlap rises to
`0.815`; route identity is no longer a clean operation address.

| regime | depth-6 MSE | in-range MSE | route outcome | decision |
|---|---:|---:|---|:---:|
| role-aligned | **1.08e-7** | **2.30e-7** | balanced, operation-specific | conditional pass |
| labels-free | 0.4633 | 0.3677 | cell-collapse/generalist route | **NO-GO** |

## Decision

**The bounded stable-cell decomposition passes conditionally, but the latent
route discovery requirement is not solved.** The role-aligned result validates
that separate memory, reusable computation, bounded state, and sparse active
execution can coexist on a controlled task. The labels-free failure prevents a
full `taklif22` GO claim: the current SGD objective does not discover the
latent computational language by itself.

The next experiment should replace the human-readable role loss with an
unsupervised usage-prototype/contrastive objective and retain the same bounded
controls. Do not move to Qwen or larger capacity until labels-free route
specialization passes.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_polynomial_operation_swap_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_polynomial_no_role_seed2026.json`
