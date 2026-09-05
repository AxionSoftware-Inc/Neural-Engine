# V0.120 `taklif22` Labels-Free Non-Polynomial Route

## Question

Does the labels-free latent route survive when the stable polynomial cell is
removed and the computation cells must learn bounded non-polynomial rules?

## Protocol

This keeps the V0.119 machine and self-supervised route objective, but changes
the hidden scalar program to four bounded transitions: shifted `tanh`, a
sinusoid-plus-affine transform, an affine sign-changing transform, and a
`tanh(x^2)` transform. The machine uses an external 32-entry fact table,
8 working-memory slots, four computation cells, top-1 execution, and an MLP
cell (`1 -> 32 -> 1`) rather than the diagnostic `1,x,x^2` basis.

The role cross-entropy remains disabled (`role_loss_weight=0`). Only state
supervision plus within-token route consistency and inter-group/global route
balance are used. Evaluation uses fixed paired cases at depths 2--6, a
fact-table swap, and a zero-memory intervention.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-nonlinear --steps 4000 --structured-value-lane --structured-cell-kind mlp --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --log-every 1000 --seed 2026 --output results/runs/latent_computational_machine_v1_bounded_nonlinear_mlp_proto_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-nonlinear --steps 4000 --structured-value-lane --structured-cell-kind mlp --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --log-every 1000 --seed 2027 --output results/runs/latent_computational_machine_v1_bounded_nonlinear_mlp_proto_seed2027.json
```

## Results

| seed | depth-6 MSE | fact-table B MSE | zero-memory MSE | route usage range | decision |
|---:|---:|---:|---:|---:|:---:|
| 2026 | **2.03e-6** | **2.09e-6** | 6.67e-4 | 24.7--25.5% | pass |
| 2027 | **4.84e-6** | **5.33e-6** | 6.98e-4 | 24.6--25.5% | pass |

In-range MSE is `4.72e-6` and `7.89e-6`. Same-program route Jaccard is 1.00
for both seeds, while different-program overlap is about 0.641--0.644. The
machine stores 60,197 parameters, while one active MLP cell contains only 97
parameters (`0.16%` of the stored model).

## Interpretation

This is a stronger signal than V0.119: the route objective is not dependent
on the exact polynomial basis, and the active-cell computation remains stable
through depth 6 on a bounded non-polynomial task. The fact-table swap remains
accurate, while zeroing external memory produces a clear degradation, so the
fact path is causally used.

This is still not a general Neural Engine result. The state is one-dimensional,
fact retrieval is a direct table lookup, and the route regularizer uses the
synthetic repeated-token grouping. Operation-swap adaptation is not counted as
a pass here because the current diagnostic adapts cell 0, while labels-free
routing does not guarantee that cell 0 owns the changed operation.

## Decision

**Conditional GO.** Proceed to a multi-dimensional state and learned memory
lookup gate using the same labels-free route objective. Do not claim a Qwen
transfer or language-model benefit until that gate passes.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_nonlinear_mlp_proto_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_nonlinear_mlp_proto_seed2027.json`
