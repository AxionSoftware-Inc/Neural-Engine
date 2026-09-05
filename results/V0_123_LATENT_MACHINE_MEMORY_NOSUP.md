# V0.123 `taklif22` Task-Only Learned Memory

## Question

Does learned external-memory addressing emerge from the task/state signal
alone, without an explicit entity-address loss?

## Protocol

This repeats V0.122's bounded two-dimensional state task, learned softmax
entity-to-row memory, four top-1 computation cells, and labels-free route
objective. The only change is `memory-supervision-weight=0`: no loss is given
for the correct fact row. The external fact values remain mutable and are
reversed for the paired fact-table evaluation.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --memory-mode learned --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_nosup_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --memory-mode learned --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2027 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_nosup_seed2027.json
```

## Results

| seed | depth-6 MSE | fact-table B MSE | zero-memory MSE | memory accuracy | route usage range | decision |
|---:|---:|---:|---:|---:|---:|:---:|
| 2026 | **3.10e-6** | **3.66e-6** | 3.04e-4 | **1.00** | 24.7--25.5% | pass |
| 2027 | **2.57e-6** | **3.26e-6** | 3.38e-4 | **1.00** | 24.8--25.5% | pass |

In-range MSE is `7.41e-6` and `4.59e-6`. Retrieval accuracy is 1.00 at
every reported depth, on both fact tables, and under the zero-memory
intervention, despite the memory address loss being exactly zero. Route
usage remains balanced and same-program route Jaccard remains 1.00.

## Interpretation

The task/state gradient is sufficient to identify the correct learned memory
row in this fixed 32-entity inventory. The fact-table swap preserving accuracy
shows that the learned address is independent of the fact content. Zeroing
the values still degrades the task, confirming that retrieval is not a
decorative side channel.

This is a strong synthetic result, but it does not yet prove open-world
addressing: the query and row keys are learned embeddings tied to the 32
entity slots. The next gate increases the inventory to 128 entities and
checks whether convergence and active-cell behavior remain stable.

## Decision

**Conditional GO.** Remove explicit memory-address supervision from the
default route/memory prototype. Keep the 128-entity scale test as the next
gate before language-model transfer.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_nosup_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_nosup_seed2027.json`
