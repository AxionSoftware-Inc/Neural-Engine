# V0.122 `taklif22` Learned External Memory

## Question

Can the machine replace the direct external-table index with a learned
entity-query-to-fact-row address while preserving multidimensional routing?

## Protocol

This is V0.121's bounded two-dimensional task and labels-free computation
route, but `memory-mode learned` replaces `fact_table[entity_id]` with a
softmax lookup over learned entity queries and learned external-row keys. The
fact values themselves remain outside the model and are swapped at
evaluation. A separate entity-address cross-entropy is used only to train and
measure the memory address; it is not an operation-role label.

The machine has four top-1 computation cells, eight working-memory slots, and
an MLP `2 -> 32 -> 2` per cell. Route role labels remain disabled. Evaluation
checks depth 2--6, fact-table swap, zero-memory intervention, and learned
memory retrieval accuracy.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --memory-mode learned --memory-supervision-weight 1.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --memory-mode learned --memory-supervision-weight 1.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2027 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_seed2027.json
```

## Results

| seed | depth-6 MSE | fact-table B MSE | zero-memory MSE | memory accuracy | route usage range | decision |
|---:|---:|---:|---:|---:|---:|:---:|
| 2026 | **1.66e-6** | **1.70e-6** | 3.03e-4 | **1.00** | 24.7--25.5% | pass |
| 2027 | **1.16e-6** | **1.34e-6** | 3.36e-4 | **1.00** | 24.8--25.5% | pass |

In-range MSE is `4.27e-6` and `3.52e-6`. Memory retrieval accuracy is 1.00
at every reported depth, on both fact tables, and under the zero-memory
intervention. The machine stores 62,569 parameters; one active computation
cell contains 162 parameters (`0.26%` of stored parameters).

Reversing the fact values leaves the learned addresses unchanged and preserves
accuracy, while zeroing the external values causes a clear degradation. This
separates learned addressing from fact content and confirms that the external
memory path is causally used.

## Decision

**Conditional GO.** Learned external addressing works together with the
multidimensional latent route on two seeds. This is not yet a claim that
memory can self-organize without auxiliary address training: the current gate
uses entity-address cross-entropy. The next control removes that auxiliary
loss, then tests more entities and held-out address layouts before any Qwen
transfer.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_seed2027.json`
