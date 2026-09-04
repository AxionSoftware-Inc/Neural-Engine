# V0.81 Per-Address Factor Mix

## Question

The accepted operation-bank model uses a shared two-coefficient factor mix for
all virtual circuit addresses. This is parameter-efficient, but it may limit
the bank's ability to specialize routes. This ablation gives each virtual
address its own two factor-mixing coefficients while retaining rank-8 adapters
and operation-specific banks.

## Reproduction

Base commit: `1780aee` (`validate rank eight operation banks`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_per_address_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_per_address_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_per_address_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_per_address_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_per_address_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_per_address_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 79.3457% | 89.5508% | 69.1406% | 7,503,009 | 1,859,520 |
| 18 | 100.0000% | 77.3926% | 87.0117% | 67.7734% | 7,503,009 | 1,859,520 |
| **mean** | **100.0000%** | **78.3691%** | 88.2813% | 68.4570% | — | — |

The shared-mix rank-8 V0.79 screen reaches 79.0771% mean at the same 3,000
steps. Per-address mixing loses 0.7080 points while adding 141,594 stored
parameters and only 32 parameters to the active estimate under the current
accounting. The additional address freedom makes optimization less stable and
does not repair the hard `multiply -> add` order.

## Decision

Rejected as the default. Keep the shared factor mix: it provides better
generalization and lower storage. The current architecture should spend extra
capacity on operation-specific banks, not on unconstrained per-address mix
coefficients.
