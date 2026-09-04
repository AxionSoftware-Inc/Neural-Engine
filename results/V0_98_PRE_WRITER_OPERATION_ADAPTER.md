# V0.98 Pre-Writer Operation Adapter

## Question

The V0.80 model applies the operation-conditioned write adapter after the
shared register writer. Because the next operation reads the resulting state,
that placement may expose an operation-specific encoding across the
composition boundary. V0.98 moves the same rank-8 adapter before the shared
writer, on `query + circuit_delta`, without adding parameters or changing the
sparse route budget.

## Reproduction

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_pre_writer_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_pre_writer_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_pre_writer_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_pre_writer_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_pre_writer_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_pre_writer_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| variant | seed 17 held-out | seed 18 held-out | two-seed mean | add -> multiply | multiply -> add | total params | active estimate |
|---|---:|---:|---:|---:|---:|---:|---:|
| pre-writer adapter | 76.56% | 79.00% | 77.78% | 86.38% | 69.19% | 7,361,415 | 1,859,488 |
| V0.80 post-state reference | 80.27% | 78.91% | **79.59%** | **90.58%** | 68.60% | 7,361,415 | 1,859,488 |

Both seeds reached 100% training accuracy. Moving the adapter before the
writer improves `multiply -> add` by only 0.59 points, but reduces
`add -> multiply` by 4.20 points and lowers the mean by 1.81 points.

## Decision

Reject pre-writer placement. The shared writer does not recover the lost
operation-specific computation, and the placement change does not create a
better reusable state interface. Keep it as an optional ablation only; do not
scale this variant to 500M or larger.
