# V0.87 Rank-16 Read Adapter Long Training

## Question

V0.86 showed a modest 3,000-step improvement from a rank-16
operation-conditioned accumulator read adapter. This confirmation runs the
same model and data for 9,000 steps to test whether that gain survives longer
optimization.

## Reproduction

Base source state: `c81df1e` (`reject circuit input normalization`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_rank16_broad_values.yaml --steps 9000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_rank16_broad_values_seed17_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_rank16_broad_values_seed17_9000.pt --examples-per-task 1024 --log-every 1000
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_rank16_broad_values.yaml --steps 9000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_rank16_broad_values_seed18_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_operation_read_adapter_rank16_broad_values_seed18_9000.pt --examples-per-task 1024 --log-every 1000
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add |
|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 78.1250% | 87.2070% | 69.0430% |
| 18 | 100.0000% | 78.9063% | 88.1836% | 69.6289% |
| **mean** | **100.0000%** | **78.5156%** | **87.6953%** | **69.3359%** |

The V0.80 rank-8 operation-bank baseline reaches 79.5898% mean at 9,000
steps. The rank-16 read adapter loses 1.0742 points overall. It preserves a
small gain on `multiply -> add` (+0.7324 points over V0.80), but lowers
`add -> multiply` enough to make the combined score worse.

## Decision

Rejected as a stable default. Longer training removes the 3,000-step overall
gain, so increasing generic operation-read rank is not the right scaling
solution. The next experiment should expose predecessor-operation identity as
explicit composition context, rather than transforming every accumulator in
the same way.

The full test suite passed after implementation: `88 passed, 2 warnings`.
