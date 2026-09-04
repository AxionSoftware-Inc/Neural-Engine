# V0.77 Operation Banks plus Numeric Scratch State

## Question

V0.70 found that a 16D learned numeric scratch state gave only a small gain
with the shared circuit bank. V0.75 found a much larger gain from separating
the circuit parameters by primitive operation. This ablation checks whether
the two mechanisms become complementary once each operation has its own bank.

## Reproduction

Base commit: `20f4617` (`validate operation circuit banks at 9000 steps`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_numeric_state16_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_numeric_state16_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_numeric_state16_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_numeric_state16_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_numeric_state16_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_numeric_state16_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 75.9766% | 82.7148% | 69.2383% | 7,407,015 | 1,905,088 |
| 18 | 100.0000% | 80.7129% | 91.3086% | 70.1172% | 7,407,015 | 1,905,088 |
| **mean** | **100.0000%** | **78.3447%** | 87.0117% | 69.6777% | — | — |

The operation-bank-only V0.75 screen has a 78.1982% mean at the same 3,000
steps. The numeric scratch state adds only 0.1465 points while adding 8,736
stored and active parameters. It does not improve the operation-bank result
reliably, and remains below the 79.5410% operation-bank mean at 9,000 steps.

## Decision

Rejected as an additional default component. Keep the operation-specific
circuit banks, typed-write interface, and learned shared state, but do not add
the numeric scratch module at this stage. The next ablation should simplify
the operation-bank model by removing redundant operation adapters and measure
whether the bank separation alone carries the gain.
