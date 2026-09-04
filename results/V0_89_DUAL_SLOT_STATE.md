# V0.89 Dual-Slot State Writer

## Question

The preceding metadata and read-adapter tests did not solve composition
transfer. This ablation changes the persistent accumulator representation:
the 384D state is split into two 192D slots, and each slot has an independent
writer. The circuit/query path still sees the concatenated state, but the
writer cannot freely mix the two slots through one dense update.

## Reproduction

Base source state: `c81df1e` (`reject circuit input normalization`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_dual_slot_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_dual_slot_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_dual_slot_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_dual_slot_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_dual_slot_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_dual_slot_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 76.8066% | 85.4492% | 68.1641% | 7,213,959 | 1,712,032 |
| 18 | 100.0000% | 76.1719% | 82.9102% | 69.4336% | 7,213,959 | 1,712,032 |
| **mean** | **100.0000%** | **76.4893%** | **84.1797%** | **68.7988%** | — | — |

The V0.79 flat-writer rank-8 operation-bank baseline reaches 79.0771% mean
at the same 3,000-step budget. Dual-slot writing loses 2.5879 points overall
and reduces `add -> multiply`; the hard order remains effectively unchanged.
It also uses fewer parameters, so this is not a capacity advantage being
masked by a larger model.

## Decision

Rejected. The shared dense writer's cross-slot mixing is useful; forcing a
separation between persistent slots discards information needed by the next
operation. The main candidate remains the flat state with operation-specific
circuit banks and rank-8 adapters.

The full test suite passed after implementation: `90 passed, 2 warnings`.
