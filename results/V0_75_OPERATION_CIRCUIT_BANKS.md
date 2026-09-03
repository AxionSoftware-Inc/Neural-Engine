# V0.75 Operation-Specific Circuit Banks

## Question

The shared circuit bank had to represent add, subtract, and multiply through
one set of sparse low-rank blocks. The operation adapters changed the query and
write interfaces, but the same routed circuit parameters were still reused by
all primitives. This may entangle primitive computation with the preceding
composition order.

This experiment gives each primitive operation its own factorized sparse
circuit bank. The hierarchical router, value/state encoder, operation adapter,
typed-write adapter, and route query remain shared. At each executed step, only
the bank for the current primitive operation is evaluated; unused operation
banks are not on the active path.

## Reproduction

Base source state: `ff0ae99` (`reject operation step routing ablation`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 77.0020% | 85.2539% | 68.7500% | 7,398,279 | 1,896,352 |
| 18 | 100.0000% | 79.3945% | 90.2344% | 68.5547% | 7,398,279 | 1,896,352 |
| **mean** | **100.0000%** | **78.1982%** | 87.7441% | 68.6523% | — | — |

The V0.69 shared-bank typed-write baseline is 69.1650% mean on the same
3,000-step broad-value screen. Operation-specific banks improve it by 9.0332
points. The gain is consistent across both seeds and is concentrated in the
held-out `add -> multiply` order; `multiply -> add` remains the harder case.

The stored parameter count rises from 3,495,299 to 7,398,279 because three
operation banks are retained. The active estimate rises only from 1,693,584 to
1,896,352 because a two-operation program touches two banks, and the active
fraction falls from 48.45% to 25.63%. This is aligned with the Neural Engine
goal of storing more capacity while executing only the operation-relevant
subset.

## Decision

Accepted as the strongest current architectural direction, but not yet as a
final default. The two-seed 3,000-step screen is a meaningful positive signal;
the next validation is a longer 9,000-step run and then a deeper-composition
test. The operation-specific bank mechanism should scale without a separate
hand-tuned routing rule for each model size because bank selection is tied to
the primitive operation vocabulary, not to the parameter count.

The full test suite passed after the implementation: `84 passed, 2 warnings`
(the existing Transformer nested-tensor warnings).
