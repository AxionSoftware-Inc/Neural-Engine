# V0.134 `taklif22` Shared Learned Key Projection

## Question

Does an external content-key memory remain stable when all keys pass through a
single trainable projection, instead of using raw dot-product matching?

## Protocol

This repeats V0.133 with 2048 external fact rows, only the first 1536 exposed
during training, and 512 completely held out. The external key dimension is
64; a single shared bias-free linear projection maps every fact key and query
key to the 32-dimensional latent space before normalized matching at
temperature `0.05`. No row-specific address parameters are created. The
bounded-vector computation path, labels-free route losses, and one-active-cell
policy are unchanged.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 2048 --train-entity-count 1536 --memory-mode content-projected --memory-key-dim 64 --memory-temperature 0.05 --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_content_projected_memory_2048e_train1536_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 2048 --train-entity-count 1536 --memory-mode content-projected --memory-key-dim 64 --memory-temperature 0.05 --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2027 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_content_projected_memory_2048e_train1536_seed2027.json
```

## Results

| seed | full depth-6 MSE | held-out MSE | fact-table B MSE | held-out retrieval | stored params | active cell |
|---:|---:|---:|---:|---:|---:|---:|
| 2026 | **1.20e-6** | **4.51e-6** | **1.35e-6** | **100%** | 62,569 | 162 |
| 2027 | **1.03e-6** | **4.30e-6** | **1.08e-6** | **100%** | 62,569 | 162 |

Route usage remains close to uniform across all four cells. The projection
adds 2,048 shared weights, while the active computation cell remains only 162
parameters.

## Interpretation

The shared learned projection preserves the V0.133 result and generalizes to
keys from rows never sampled during training. This is stronger than a
per-entity address bank and remains compatible with sparse computation. The
experiment still uses externally paired query/fact keys, so it is not yet a
semantic key encoder or an end-to-end memory-learning result.

## Decision

**GO for shared key transforms; reject row-specific learned address banks as
the default.** The next meaningful gate is to generate query and fact keys
from separate external content representations, with a controlled matching
task, before connecting this memory interface to Qwen-derived FFN circuits.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_content_projected_memory_2048e_train1536_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_content_projected_memory_2048e_train1536_seed2027.json`
