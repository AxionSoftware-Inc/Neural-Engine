# V0.86 Rank-16 Operation-Conditioned Read Adapter

## Question

V0.85's rank-8 operation-conditioned read adapter moved the difficult
`multiply -> add` order only marginally. This screen doubles the read-path
rank to 16 while keeping the operation-specific circuit banks, rank-8 query
adapter, rank-8 write adapter, shared factorized router, and all data settings
unchanged.

The hypothesis is that the common accumulator needs a wider operation-local
input transform before pair formation, not a separate router or a forced
active-path size.

## Reproduction

Base source state: `c81df1e` (`reject circuit input normalization`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_rank16_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_rank16_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_rank16_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_rank16_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_rank16_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_rank16_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 80.2734% | 91.0156% | 69.5313% | 7,399,431 | 1,897,504 |
| 18 | 100.0000% | 79.3457% | 88.4766% | 70.2148% | 7,399,431 | 1,897,504 |
| **mean** | **100.0000%** | **79.8096%** | **89.7461%** | **69.8730%** | — | — |

The V0.79 full-query rank-8 operation-bank baseline reaches 79.0771% mean
at the same 3,000-step budget. Rank 16 improves the mean by 0.7324 points
and improves `multiply -> add` over V0.80's 68.6035% mean by 1.2695 points.
It adds 38,016 stored parameters and about 38,016 to the active estimate.

## Decision

Accepted as a provisional candidate, not yet as the final default. The gain
is modest but consistent across both seeds and directly targets the hard
composition order. Run the same configuration for 9,000 steps before scaling
the model capacity further. If the gain survives, this becomes the state
interface for future 700M/1B capacity experiments.

The full test suite passed after implementation: `88 passed, 2 warnings`.
