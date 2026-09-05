# V0.119 `taklif22` Labels-Free Latent Route

## Question

Can the bounded stable-cell machine discover reusable latent computational
routes without human-readable operation-role labels?

## Protocol

This keeps the V0.118 bounded synthetic machine: external 32-entry fact table,
8 working-memory slots, four computation cells, top-1 active execution, and
the stable `1,x,x²` diagnostic cell basis. Training uses 2--4 steps and fixed
paired evaluation cases cover depths 2--6, fact-table swap, and zero-memory
intervention.

The role cross-entropy is disabled (`role_loss_weight=0`). Instead, the only
route regularizers are self-supervised: positions carrying the same input
token are encouraged to share a route distribution, different token-group
means are discouraged from overlapping, and global usage is balanced. No
`ADD`, `MUL`, or other semantic role label is supplied to the optimizer.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded --steps 4000 --structured-value-lane --structured-cell-kind polynomial --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --log-every 1000 --seed 2026 --output results/runs/latent_computational_machine_v1_bounded_polynomial_proto2_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded --steps 4000 --structured-value-lane --structured-cell-kind polynomial --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --log-every 1000 --seed 2027 --output results/runs/latent_computational_machine_v1_bounded_polynomial_proto2_seed2027.json
```

## Results

| seed | depth-6 MSE | fact-table B MSE | zero-memory MSE | route usage range | decision |
|---:|---:|---:|---:|---:|:---:|
| 2026 | **8.62e-8** | **1.47e-7** | 0.00340 | 24.7--25.5% | pass |
| 2027 | **1.16e-7** | **1.80e-7** | 0.00327 | 24.6--25.5% | pass |

In-range MSE is `1.85e-7` and `2.21e-7` respectively. Same-program route
Jaccard is 1.00 for both seeds, while different-program overlap is about
0.641--0.643. One of four cells is active per program step; each active
polynomial cell has only three learned coefficients, and the full machine has
59,821 stored parameters.

The earlier consistency/balance formulation failed because it pushed every
operation distribution toward uniform mixing and produced non-finite/unusable
routes. The revised global-balance plus inter-group-overlap penalty avoids that
collapse in both seeds.

## Decision

**Conditional GO for the bounded synthetic latent-route stage.** This is the
first two-seed result where the machine learns operation-reusable routes
without role labels, preserves mutable facts, uses working memory, and remains
accurate through depth 6 with only one active cell.

The result is not yet a general Neural Engine or language-model result. The
polynomial basis is a strong inductive bias for this bounded task, fact
retrieval is still a direct table lookup, and the route objective groups
synthetic input tokens. The next gate must use a bounded non-polynomial or
multi-dimensional state task, a learned/hierarchical memory lookup, and the
same labels-free route objective. Qwen transfer remains paused until that
generalization gate passes.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_polynomial_proto2_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_polynomial_proto2_seed2027.json`
