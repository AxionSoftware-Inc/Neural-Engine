# V0.90 500M Virtual Operation-Bank Scratch Scale

## Question

V0.79/V0.80 established operation-specific factorized circuit banks as the
best current architecture at 300M virtual capacity. This screen increases
the factorized bank from 23,600 to 39,300 virtual circuits while keeping the
active circuit count, state width, adapters, data, and optimizer unchanged.

## Reproduction

```powershell
python -u train_composition.py --config configs/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 78.8574% | 88.3789% | 69.3359% | 9,089,415 | 1,859,488 |
| 18 | 100.0000% | 78.3691% | 88.9648% | 67.7734% | 9,089,415 | 1,859,488 |
| **mean** | **100.0000%** | **78.6133%** | **88.6719%** | **68.5547%** | — | — |

The V0.79 300M operation-bank baseline reaches 79.0771% mean at the same
3,000-step budget. Adding 15,700 virtual circuits lowers the mean by 0.4639
points and leaves the hard order unchanged. The stored parameter count rises
from 7,361,415 to 9,089,415, while the estimated active path remains
1,859,488.

## Decision

Scratch capacity scaling is rejected as a quality improvement. The new rows
are not automatically converted into useful reusable computation; the
300M architecture remains the better short-budget model. A parent-growth
control is recorded separately in V0.91.
