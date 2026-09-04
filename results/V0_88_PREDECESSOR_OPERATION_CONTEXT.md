# V0.88 Explicit Predecessor-Operation Context

## Question

The difficult held-out orders are unseen operation compositions. The current
query contains the current operation and step, but not the preceding
primitive explicitly. This ablation adds a four-entry predecessor-operation
embedding, including a START entry, to the query before routing and writing.

The intention was to make the composition interface explicit without forcing
an active-path size or changing the shared accumulator format.

## Reproduction

Base source state: `c81df1e` (`reject circuit input normalization`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_predecessor_context_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_predecessor_context_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_predecessor_context_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_predecessor_context_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_predecessor_context_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_predecessor_context_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 74.4629% | 80.2734% | 68.6523% | 7,362,951 | 1,861,024 |
| 18 | 100.0000% | 77.6367% | 86.4258% | 68.8477% | 7,362,951 | 1,861,024 |
| **mean** | **100.0000%** | **76.0498%** | **83.3496%** | **68.7500%** | — | — |

The V0.79 full-query rank-8 operation-bank baseline reaches 79.0771% mean at
the same 3,000-step budget. Explicit predecessor context loses 3.0273 points
overall. It moves `multiply -> add` only to 68.7500%, essentially unchanged,
while reducing `add -> multiply`.

## Decision

Rejected. The model does not need more operation metadata in the query; it
needs a state representation and composition interface that preserves the
right reusable information across an operation boundary. Future work should
test structured multi-slot state or dual-path value/carry state, not more
router metadata.

The full test suite passed after implementation: `89 passed, 2 warnings`.
