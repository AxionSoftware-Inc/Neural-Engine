# V0.84 Operation-Specific Router-Key Residuals

## Question

Operation-specific circuit banks improved the 300M virtual-capacity model,
but the shared factorized router still chooses factor addresses for all
primitives. This ablation adds a zero-initialized, operation-conditioned
residual to the shared factor keys. The base factor geometry remains shared,
while each operation can learn a local routing adjustment.

## Reproduction

Base source state: `c81df1e` (`reject circuit input normalization`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_router_keys_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_router_keys_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_router_keys_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_router_keys_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_router_keys_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_router_keys_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 74.3652% | 81.3477% | 67.3828% | 7,538,823 | 1,871,776 |
| 18 | 100.0000% | 76.5625% | 85.5469% | 67.5781% | 7,538,823 | 1,871,776 |
| **mean** | **100.0000%** | **75.4639%** | **83.4473%** | **67.4805%** | — | — |

The V0.79 full-query rank-8 operation-bank baseline reaches 79.0771% mean at
the same 3,000-step budget. Operation-specific key residuals lose 3.6133
points overall and reduce `multiply -> add` to 67.4805%. They add 177,408
stored parameters and only about 12,288 parameters to the two-step active
estimate.

## Decision

Rejected. The shared factor geometry is an important invariant: letting each
operation move factor keys independently harms composition transfer, even
when the residual starts at zero. Keep router keys shared and focus on the
state interface entering each operation's pair/circuit computation.

The full test suite passed after implementation: `87 passed, 2 warnings`.
