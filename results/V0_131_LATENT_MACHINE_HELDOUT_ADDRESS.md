# V0.131 `taklif22` Held-Out Address Gate

## Question

Does the current learned memory generalize to entity addresses that were not
present during training?

## Protocol

The 2048-entity V0.130 reference is changed so only the first 1536 entities
are sampled during training. The remaining 512 entities are never seen by
the per-entity query/key bank. Evaluation reports the normal mixed inventory
and a separate `heldout_entity_eval` for the unseen tail. Computation remains
bounded-vector, four top-1 MLP cells, labels-free route consistency/balance,
32 hierarchical buckets, and address guidance weight 0.1.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 2048 --train-entity-count 1536 --memory-mode hierarchical --memory-buckets 32 --memory-supervision-weight 0.1 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_hierarchical32_memory_weight01_2048e_train1536_seed2026.json
```

## Results

| split | MSE | memory accuracy | route usage | decision |
|---|---:|---:|---:|:---:|
| mixed 2048 inventory | 1.32e-4 | 75.3% | balanced | partial |
| held-out 512 entities | **1.46e-2** | **0.0%** | balanced | fail |

For comparison, the same configuration trained on all 2048 entities reaches
depth-6 MSE `1.11e-6` and `9.23e-7` across two seeds with 100% retrieval.
The held-out failure is therefore not a general training instability.

## Interpretation

The computation bank generalizes: route usage stays balanced and only one
cell is active. The address bank does not: learned per-entity query/key
vectors memorize the training inventory, so unseen entity IDs have no learned
address. This identifies a real architectural boundary for the current
memory implementation.

## Decision

**Reject the fixed per-entity address bank as the final memory design.** Keep
it as a closed-world synthetic reference, but the next architecture must use
shared content/coordinate encoders or an external key supplied with each fact
so unseen entities can be addressed without allocating a new learned vector.
Do not transfer this memory design to Qwen yet.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_hierarchical32_memory_weight01_2048e_train1536_seed2026.json`
