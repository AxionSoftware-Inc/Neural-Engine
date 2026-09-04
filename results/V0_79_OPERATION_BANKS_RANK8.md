# V0.79 Operation Banks with Rank-8 Adapters

## Question

V0.75/V0.76 use rank-16 operation query and typed-write adapters together
with operation-specific circuit banks. This sweep halves both adapter ranks to
8 while leaving the circuit bank, router, state, and training budget unchanged.
The goal is to reduce active cost without sacrificing the broad-value gain.

## Reproduction

Base commit: `5b4a609` (`attribute operation bank gain`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 81.1035% | 92.9688% | 69.2383% | 7,361,415 | 1,859,488 |
| 18 | 100.0000% | 77.0508% | 85.8398% | 68.2617% | 7,361,415 | 1,859,488 |
| **mean** | **100.0000%** | **79.0771%** | 89.4043% | 68.7500% | — | — |

The rank-16 V0.75 screen reaches 78.1982% mean at the same 3,000-step
budget. Rank 8 improves the mean by 0.8789 points while reducing total and
estimated active parameters by 36,864. The improvement is a positive but
moderate signal; the longer-budget rank-8 confirmation is still required.

## Decision

Advance rank 8 as the current efficiency candidate. Keep rank 16 as the
quality reference until the 9,000-step comparison is complete. This sweep
supports the broader design principle: operation-specific sparse capacity is
more valuable than simply widening every operation-conditioned adapter.
