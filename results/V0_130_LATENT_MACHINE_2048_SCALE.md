# V0.130 `taklif22` 2048-Entity Capacity Gate

## Question

Does the current reference configuration keep exact learned memory retrieval
and one-active-cell computation when the external inventory grows to 2048
entities?

## Protocol

This uses the promoted V0.129 configuration: bounded two-dimensional state,
32 hierarchical memory buckets, 0.1 address guidance, four top-1 MLP cells,
and labels-free operation routing. Both seeds use 4,000 training steps and
fixed depth 2--6, fact-table swap, and zero-memory evaluations.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 2048 --memory-mode hierarchical --memory-buckets 32 --memory-supervision-weight 0.1 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_hierarchical32_memory_weight01_2048e_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 2048 --memory-mode hierarchical --memory-buckets 32 --memory-supervision-weight 0.1 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2027 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_hierarchical32_memory_weight01_2048e_seed2027.json
```

## Results

| seed | depth-6 MSE | fact-table B MSE | in-range MSE | zero-memory MSE | memory accuracy | route usage | decision |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2026 | **1.11e-6** | **1.33e-6** | 3.82e-6 | 2.68e-4 | **100%** | balanced | pass |
| 2027 | **9.23e-7** | **1.06e-6** | 4.95e-6 | 2.65e-4 | **100%** | balanced | pass |

The machine stores 192,617 parameters, dominated by the learned external
memory bank, while exactly one vector computation cell remains active with
162 parameters (`0.084%` of stored parameters). Reversing fact values
preserves accuracy, and zeroing them causes a clear degradation.

## Interpretation

The promoted memory/routing configuration scales from 32 to 2048 rows in this
synthetic gate without retrieval collapse or active-cell growth. This is
strong evidence for conditional computation in the computation bank, but it
is not a billion-parameter language-model result: the entity query/key bank
still allocates per-inventory embeddings and address guidance is present.

## Decision

**GO to the next generalization gate.** Keep this as the current synthetic
reference. Before Qwen transfer, remove the fixed entity-embedding assumption
with content-addressed/unseen-entity evaluation, then test longer programs
and a larger cell bank.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_hierarchical32_memory_weight01_2048e_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_hierarchical32_memory_weight01_2048e_seed2027.json`
