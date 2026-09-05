# V0.121 `taklif22` Multidimensional Latent Route

## Question

Does the labels-free latent route remain stable when a computation cell must
update a coupled two-dimensional state instead of a scalar register?

## Protocol

The bounded-vector task uses four coupled transitions. The first is a shifted
`tanh` mixing both coordinates, the second combines affine and sinusoidal
cross-coordinate terms, the third is a sign-changing rotation-like update,
and the fourth is a bounded quadratic mixing rule. The state stays bounded,
but no single-coordinate polynomial shortcut is available.

The machine has an external 32-entry fact table with two values per entity,
8 working-memory slots, four computation cells, and top-1 execution. Each
cell is an MLP `2 -> 32 -> 2`. Role cross-entropy is disabled; training uses
state supervision plus the V0.119 labels-free route consistency and balance
objective.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_mlp_proto_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2027 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_mlp_proto_seed2027.json
```

## Results

| seed | depth-6 MSE | fact-table B MSE | zero-memory MSE | route usage range | decision |
|---:|---:|---:|---:|---:|:---:|
| 2026 | **1.52e-6** | **1.61e-6** | 3.02e-4 | 24.7--25.5% | pass |
| 2027 | **1.27e-6** | **1.46e-6** | 3.36e-4 | 24.8--25.5% | pass |

In-range MSE is `3.92e-6` and `4.11e-6`. Same-program route Jaccard is 1.00
for both seeds, while different-program overlap is about 0.641--0.644. The
machine stores 60,521 parameters; one active vector MLP cell contains 162
parameters (`0.27%` of the stored model).

Zeroing the external fact table causes a two-order-of-magnitude degradation,
while reversing the fact table preserves accuracy. This supports causal use
of the external fact path and shows that the route is not simply memorizing
the fixed evaluation table.

## Decision

**Conditional GO.** The route mechanism survives scalar-to-vector expansion
and coupled non-polynomial transitions on two seeds. The next gate is learned
addressing of the external fact table: the model must retrieve by an entity
query rather than receive the row through a direct Python index. Only after
that gate should we consider transfer to larger Neural Engine cells or Qwen.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_mlp_proto_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_mlp_proto_seed2027.json`
