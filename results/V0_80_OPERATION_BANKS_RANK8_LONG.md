# V0.80 Rank-8 Operation Banks at 9,000 Steps

## Purpose

V0.79 found that rank-8 operation adapters improve the 3,000-step screen while
using fewer active parameters. This follow-up compares rank 8 and rank 16 at
the longer 9,000-step budget, where the operation-specific circuit-bank gain
has already been validated.

## Reproduction

Base commit: `dd3535c` (`screen rank eight operation adapters`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values.yaml --steps 9000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed17_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed17_9000.pt --examples-per-task 1024 --log-every 1000
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values.yaml --steps 9000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed18_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed18_9000.pt --examples-per-task 1024 --log-every 1000
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 80.2734% | 92.0898% | 68.4570% | 7,361,415 | 1,859,488 |
| 18 | 100.0000% | 78.9063% | 89.0625% | 68.7500% | 7,361,415 | 1,859,488 |
| **mean** | **100.0000%** | **79.5898%** | 90.5762% | 68.6035% | — | — |

The rank-16 V0.76 reference reaches 79.5410% mean at 9,000 steps with
1,896,352 active parameters. Rank 8 is statistically tied in this two-seed
screen (+0.0488 points) while saving 36,864 active and stored parameters. It
also improves the 3,000-to-9,000-step mean from 79.0771% to 79.5898%.

## Decision

Accept rank 8 as the current efficiency default and retain rank 16 only as a
quality comparison checkpoint. The central architecture is now operation-
specific sparse circuit banks plus low-rank operation read/write adapters. The
remaining bottleneck is the asymmetric `multiply -> add` order, not adapter
width or raw capacity.
