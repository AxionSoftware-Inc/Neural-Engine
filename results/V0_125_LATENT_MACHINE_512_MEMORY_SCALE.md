# V0.125 `taklif22` 512-Entity Memory Scale Failure

## Question

Why does task-only learned addressing that succeeds at 32 and 128 entities
fail when the external memory inventory reaches 512 rows?

## Protocol

The task-only configuration is V0.124: bounded two-dimensional state, four
top-1 MLP computation cells, labels-free route consistency/balance, learned
entity-query memory, and zero explicit memory-address loss. Two seeds are
run at 512 entities. A 2026 control then restores the explicit address loss
to distinguish optimization/objective failure from representational failure.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 512 --memory-mode learned --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_nosup_512e_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 512 --memory-mode learned --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2027 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_nosup_512e_seed2027.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 512 --memory-mode learned --memory-supervision-weight 1.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_sup_512e_seed2026.json
```

## Results

| condition | seed | depth-6 MSE | fact-table B MSE | memory accuracy | route usage | decision |
|---|---:|---:|---:|---:|---:|:---:|
| task-only | 2026 | 2.89e-4 | 4.44e-4 | 40.9% | balanced | fail |
| task-only | 2027 | 2.41e-4 | 2.97e-4 | 42.6% | balanced | fail |
| address-supervised control | 2026 | **7.45e-7** | **9.34e-7** | **100%** | balanced | recover |

The task-only route remains balanced in all three runs and one cell remains
active, so the failure is not route collapse. At 512 rows the task loss does
not provide a sufficiently sharp address signal: retrieval stays near a
coarse partial match and the fact swap no longer preserves a low error. The
explicit address objective recovers exact retrieval and even beats the
32/128-row task-only quality, showing that the memory/query representation has
enough capacity.

## Decision

**Reject task-only O(N) address scaling at 512. Keep the learned-cell route,
but do not rely on raw task gradients to train a large row-wise memory.** The
next architecture experiment should use hierarchical/content-addressed
lookup or a contrastive address objective with sampled negatives. This is an
optimization/objective bottleneck, not evidence that the active-cell
architecture itself has failed.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_nosup_512e_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_nosup_512e_seed2027.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_learned_memory_sup_512e_seed2026.json`
