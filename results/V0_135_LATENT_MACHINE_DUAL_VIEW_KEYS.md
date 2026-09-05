# V0.135 `taklif22` Dual-View External Keys

## Question

Can a shared sparse memory interface learn to align separate query-key and
fact-key representations, rather than relying on exact externally paired
vectors?

## Protocol

The experiment uses 2048 external fact rows. Only the first 1536 entity rows
are sampled during training; the final 512 are held out. A random external
key table is used as the fact view. The query view is produced by applying a
fixed random orthogonal transform to the same latent key table, so query and
fact vectors are different but preserve a row-level relation. The model uses
two global bias-free projections (`content-dual-projected`) to align the
views. These projections are shared across every row; no per-entity address
parameters are allocated.

Address cross-entropy is enabled only to train the alignment, while the
bounded-vector computation path keeps labels-free route consistency/balance
and one active cell per step. The 512 held-out rows test transfer of the
shared relation, not memorization of row embeddings.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 2048 --train-entity-count 1536 --memory-mode content-dual-projected --memory-key-dim 64 --memory-temperature 0.05 --memory-supervision-weight 1.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_content_dual_projected_memory_2048e_train1536_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 2048 --train-entity-count 1536 --memory-mode content-dual-projected --memory-key-dim 64 --memory-temperature 0.05 --memory-supervision-weight 1.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2027 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_content_dual_projected_memory_2048e_train1536_seed2027.json
```

## Results

| seed | full depth-6 MSE | held-out MSE | fact-table B MSE | held-out retrieval | stored params | active cell |
|---:|---:|---:|---:|---:|---:|---:|
| 2026 | **2.26e-6** | **3.84e-6** | **2.42e-6** | **100%** | 64,617 | 162 |
| 2027 | **2.22e-6** | **6.05e-6** | **2.45e-6** | **100%** | 64,617 | 162 |

Route usage stayed close to 25% per cell. The two shared projections add
4,096 weights; the active computation cell remains only 162 parameters.

## Interpretation

The model transfers a learned relation between separate query and fact views
to rows never seen during training. This is a stronger result than V0.133's
exact external-key control and V0.134's shared projection over identical
views. It supports a shared key-alignment circuit as a viable replacement for
per-entity learned address banks.

The remaining limitation is that the two key views are still synthetically
generated and paired by the benchmark. This does not yet prove semantic
addressing from natural text or Qwen hidden states. The next integration gate
should feed real frozen-FFN representations into this interface and preserve
the same held-out-content test.

## Decision

**GO for shared dual-view key alignment.** Keep row-specific address tables
rejected as the default and move the next experiment toward Qwen-derived
content features, not larger fixed embedding banks.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_content_dual_projected_memory_2048e_train1536_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_content_dual_projected_memory_2048e_train1536_seed2027.json`
