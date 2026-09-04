# V0.78 Operation Banks without Operation Adapters

## Question

V0.75/V0.76 show that separate circuit banks are the main positive direction,
but those models also retain the operation-conditioned query and typed-write
adapters. This control removes both low-rank adapters while keeping only the
operation-specific circuit banks, shared state, and shared router.

The purpose is attribution and simplification: measure how much of the gain
comes from bank separation itself and how much comes from adapter synergy.

## Reproduction

Base commit: `d0a4e21` (`reject numeric state with operation banks`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_operation_circuit_banks_no_operation_adapters_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_operation_circuit_banks_no_operation_adapters_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_operation_circuit_banks_no_operation_adapters_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_operation_circuit_banks_no_operation_adapters_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_operation_circuit_banks_no_operation_adapters_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_operation_circuit_banks_no_operation_adapters_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 75.0488% | 82.1289% | 67.9688% | 7,322,247 | 1,820,320 |
| 18 | 100.0000% | 76.2207% | 86.5234% | 65.9180% | 7,322,247 | 1,820,320 |
| **mean** | **100.0000%** | **75.6348%** | 84.3262% | 66.9434% | — | — |

The full operation-bank plus typed-write V0.75 screen reaches 78.1982% mean
at the same 3,000-step budget. Removing both adapters costs 2.5635 points,
while the result remains 6.4697 points above the V0.69 shared-bank typed-write
baseline of 69.1650%.

The control reduces stored and estimated active parameters by 76,032, but the
quality loss is larger than the efficiency benefit. The operation-specific
banks carry the majority of the improvement; the adapters provide a useful
secondary interface alignment.

## Decision

Keep both operation adapters in the current candidate architecture. This
control confirms that the central mechanism is primitive-specific circuit
capacity, but the adapters should be retained unless a lower-rank sweep shows
that most of their quality can be recovered more cheaply.
