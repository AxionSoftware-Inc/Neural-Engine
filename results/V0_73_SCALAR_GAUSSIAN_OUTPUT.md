# V0.73 Scalar Gaussian Output Ablation

## Question

The broad benchmark is numeric, so a conventional class head may discard the
ordering and distance between target values. This ablation replaces the
learned 512-way output projection with a continuous scalar readout. The scalar
is converted into class logits by a fixed Gaussian distance over class
positions, with temperature `16.0` and initial coordinate `64.0`.

This is an output-interface test only. The Dynamic Register, typed-write
adapter, sparse factorized bank, routing, and training data are unchanged.

## Reproduction

Base source state: the V0.72 compact-head commit `5cd0895`.

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_scalar_output_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_scalar_output_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_scalar_output_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_scalar_output_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_scalar_output_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_scalar_output_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 96.4983% | 38.6230% | 20.7031% | 56.5430% | 3,298,564 | 1,496,849 |
| 18 | 98.3119% | 36.3281% | 21.9727% | 50.6836% | 3,298,564 | 1,496,849 |
| **mean** | **97.4051%** | **37.4756%** | 21.3379% | 53.6133% | — | — |

The typed-write learned-head baseline is 69.1650% mean on the same 3,000-step
broad screen. The scalar output therefore loses 31.6895 points while also
slowing training convergence. Its parameter count is lower, but the efficiency
does not compensate for the quality collapse.

## Decision

Rejected. The current learned accumulator state does not expose a stable
single scalar coordinate that can support this fixed metric decoder. The
numeric bottleneck is therefore not solved by imposing an ordered output
geometry; future work must improve the internal composition state or its
operation interface first.

The scalar mode remains available as an explicit ablation, while the learned
output head stays the default. The full test suite passed after the addition:
`83 passed, 2 warnings` (the existing Transformer nested-tensor warnings).
