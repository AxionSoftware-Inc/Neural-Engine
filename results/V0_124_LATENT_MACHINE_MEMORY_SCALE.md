# V0.124 `taklif22` Learned Memory Scale Gate

## Question

Does task-only learned external addressing remain usable when the fact-table
inventory grows from 32 to 128 entities?

## Protocol

This repeats V0.123 exactly: bounded two-dimensional state, four top-1 MLP
cells, labels-free route consistency/balance, learned entity-query memory,
and no memory-address supervision. Only `num_entities` changes from 32 to
128. Evaluation uses fixed depth 2--6 cases, reversed fact values, zeroed
fact values, and explicit learned-memory retrieval accuracy.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 128 --memory-mode learned --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_nosup_128e_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 128 --memory-mode learned --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2027 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_nosup_128e_seed2027.json
```

## Results

| seed | depth-6 MSE | fact-table B MSE | zero-memory MSE | memory accuracy | route usage range | decision |
|---:|---:|---:|---:|---:|---:|:---:|
| 2026 | **6.22e-6** | **8.44e-6** | 3.12e-4 | **1.00** | 24.7--25.5% | pass |
| 2027 | **6.90e-6** | **8.37e-6** | 3.12e-4 | **1.00** | 24.6--25.5% | pass |

In-range MSE is `1.26e-5` and `9.79e-6`. Retrieval accuracy is 1.00 at
every reported depth, on both fact tables, and under zero-memory evaluation.
The 128-entity machine stores 68,713 parameters; one active cell still has
162 parameters (`0.24%` of stored parameters).

Relative to V0.123's 32-entity depth-6 MSE (`3.10e-6` and `2.57e-6`), the
128-entity scores are roughly 2.0--2.5x worse. This is a measurable scaling
cost, but not a route or retrieval collapse: fact swap, zero-memory causal
degradation, balanced usage, and exact retrieval all remain intact.

## Decision

**Conditional GO, with a scaling warning.** The learned memory mechanism
survives a 4x entity-inventory increase, but quality is not yet scale
invariant. Before moving to Qwen or claiming billion-parameter behavior, test
512 entities and a content-addressed or hierarchical memory that does not
allocate one learned key/query pair per fixed row.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_nosup_128e_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_nosup_128e_seed2027.json`
