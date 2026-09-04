# V0.93 Ordered Factor Slots

## Question

V0.92 showed that controlling when new factor rows become routable does not
fix 500M growth. A separate concern was that a factor pair `(a, b)` was
assembled from one shared factor table, potentially making ordered addresses
too similar. V0.93 gives the first and second factor slots separate reusable
factor tables. The active route still reads only the selected factor rows,
but `(a, b)` and `(b, a)` now have genuinely different circuit parameters.

## Reproduction

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_ordered_factor_slots_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_ordered_factor_slots_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_ordered_factor_slots_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_ordered_factor_slots_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_ordered_factor_slots_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_ordered_factor_slots_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 77.1973% | 85.5469% | 68.8477% | 13,215,879 | 1,859,488 |
| 18 | 99.9860% | 77.2949% | 86.7188% | 67.8711% | 13,215,879 | 1,859,488 |
| **mean** | **99.9930%** | **77.2461%** | **86.1328%** | **68.3594%** | — | — |

The ordered-slot mean is 1.8310 points below the V0.79 shared-factor 300M
baseline (79.0771%). It also takes about 1.79x the stored parameters while
the active estimate remains unchanged. The extra order-specific freedom does
not improve the held-out operation orders.

## Decision

Reject as the default. The current failure is not explained by only one
shared factor table collapsing `(a, b)` and `(b, a)`. Before another
architecture expansion, inspect route utilization and the effective number of
factor rows receiving useful gradient; the evidence now points to credit
assignment or the recurrent numeric interface rather than raw factor-bank
capacity.
