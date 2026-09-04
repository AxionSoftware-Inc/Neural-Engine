# V0.85 Operation-Conditioned Read Adapter

## Question

V0.80's operation-specific circuit banks remain the strongest default, but
the `multiply -> add` order is still weak. The shared register state may be
valid globally yet require a small operation-specific read transform before
the next operation forms its pair query. This ablation adds a rank-8
operation-conditioned residual to the accumulator only on that read path.
The written accumulator remains in the shared state space.

## Reproduction

Base source state: `c81df1e` (`reject circuit input normalization`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 79.7363% | 90.6250% | 68.8477% | 7,380,999 | 1,879,072 |
| 18 | 100.0000% | 77.4902% | 85.5469% | 69.4336% | 7,380,999 | 1,879,072 |
| **mean** | **100.0000%** | **78.6133%** | **88.0859%** | **69.1406%** | — | — |

The V0.79 full-query rank-8 operation-bank baseline reaches 79.0771% mean at
the same 3,000-step budget. The read adapter loses 0.4639 points overall,
although `multiply -> add` improves by 0.0879 points over V0.80's 68.6035%
mean. The gain is too small and inconsistent to justify the added 19,584
stored parameters.

## Decision

Rejected as the rank-8 default, but not discarded as a research direction.
The small hard-order movement suggests that an operation-typed state interface
may be relevant, while the rank-8 transform lacks enough capacity or the
current shared write format is the real bottleneck. A rank-16 read-capacity
screen is the next targeted test; if it also fails, stop expanding adapters
and move to a deeper state representation change.

The full test suite passed after implementation: `88 passed, 2 warnings`.
