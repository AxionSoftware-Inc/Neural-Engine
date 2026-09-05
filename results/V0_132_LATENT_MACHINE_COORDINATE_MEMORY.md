# V0.132 `taklif22` Shared Coordinate Memory Control

## Question

Can the held-out address failure be removed by replacing per-entity learned
query/key vectors with a shared coordinate address function?

## Protocol

The external fact table has 2048 rows, but only the first 1536 entity IDs are
used during training. The remaining 512 are held out. `memory-mode coordinate`
uses a shared deterministic coordinate kernel over row positions instead of
allocating learned query/key vectors per entity. The fact values remain
external and mutable. The computation path is unchanged: bounded 2D state,
four top-1 MLP cells, labels-free route consistency/balance, and one active
cell per step.

This is deliberately a parameter-free address control, not yet a learned
content-addressed memory. Its purpose is to test whether the fixed-entity
embedding bank itself is the held-out bottleneck.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 2048 --train-entity-count 1536 --memory-mode coordinate --memory-temperature 1.0 --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_coordinate_memory_2048e_train1536_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 2048 --train-entity-count 1536 --memory-mode coordinate --memory-temperature 1.0 --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2027 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_coordinate_memory_2048e_train1536_seed2027.json
```

## Results

| seed | full depth-6 MSE | held-out MSE | fact-table B MSE | held-out memory accuracy | stored params | decision |
|---:|---:|---:|---:|---:|---:|:---:|
| 2026 | **1.28e-6** | **3.76e-6** | **1.28e-6** | **100%** | 60,521 | pass |
| 2027 | **1.03e-6** | **4.55e-6** | **1.18e-6** | **100%** | 60,521 | pass |

The one-active vector cell still has 162 parameters (`0.27%` of stored
parameters), and route usage remains balanced. The held-out task error is
only a few times the mixed-inventory error, unlike V0.131's fixed-bank
held-out MSE of `1.46e-2` and 0% retrieval.

## Interpretation

The held-out failure was caused by the per-entity learned address bank, not by
the sparse computation route. A shared address function can retrieve unseen
rows and preserve the active-cell path with no memory-bank parameters.

This control is intentionally limited: the coordinate is derived from the
entity index, so it does not yet prove general semantic/content addressing.
The next gate should replace this deterministic coordinate with a shared
learned address encoder and externally supplied fact keys, then test shuffled
and unseen keys.

## Decision

**Conditional GO for shared addressing; reject per-entity embeddings as the
default memory design.** Continue toward learned content keys before any Qwen
transfer.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_coordinate_memory_2048e_train1536_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_coordinate_memory_2048e_train1536_seed2027.json`
