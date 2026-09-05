# V0.133 `taklif22` External Content-Key Memory

## Question

Can the memory address generalize to entity rows that were never used during
training when the address is an externally supplied content key rather than a
learned per-entity embedding or an entity coordinate?

## Protocol

The fact table has 2048 rows. Only the first 1536 entity IDs are sampled while
training; the final 512 rows are held out completely. For `memory-mode content`
the harness creates one external random key per row and supplies the same key
table as the fact-key table and the query-key table. Retrieval is normalized
dot-product matching at temperature `0.05`. The key table is not a model
parameter, is not optimized, and is not derived from the entity ID. Fact
values remain mutable external data.

The computation path is unchanged: bounded 2D state, four reusable MLP cells,
labels-free route consistency/balance, and exactly one active cell per step.
This is a retrieval/addressing control, not a claim that the model has learned
semantic keys.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 2048 --train-entity-count 1536 --memory-mode content --memory-key-dim 64 --memory-temperature 0.05 --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_content_memory_2048e_train1536_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 2048 --train-entity-count 1536 --memory-mode content --memory-key-dim 64 --memory-temperature 0.05 --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2027 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_content_memory_2048e_train1536_seed2027.json
```

## Results

| seed | full depth-6 MSE | held-out MSE | fact-table B MSE | held-out retrieval | stored params | active cell |
|---:|---:|---:|---:|---:|---:|---:|
| 2026 | **9.05e-7** | **4.21e-6** | **1.07e-6** | **100%** | 60,521 | 162 |
| 2027 | **1.32e-6** | **4.27e-6** | **1.32e-6** | **100%** | 60,521 | 162 |

Route usage stayed balanced at approximately 25% per cell. The held-out
retrieval and task error match the trained rows closely, despite the last 512
rows never appearing in a training batch.

## Interpretation

This is a strong positive signal against the fixed learned-address-bank
bottleneck seen in V0.131. It shows that a shared content-key lookup can keep
the sparse computation path intact and handle unseen rows without allocating
per-row model parameters. It does **not** yet show semantic addressing: the
keys are externally supplied and query keys exactly match fact keys. A next
validation should independently permute the key/table presentation and add a
shared learned key encoder before treating this as a production memory design.

## Decision

**GO for external/shared addressing; reject fixed per-entity address banks as
the default.** Keep the computation-cell architecture unchanged and continue
with a shared key-encoding/generalization gate before attempting Qwen-derived
FFN transfer.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_content_memory_2048e_train1536_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_content_memory_2048e_train1536_seed2027.json`
