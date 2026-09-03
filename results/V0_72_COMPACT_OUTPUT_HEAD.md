# V0.72 Compact Output Head Ablation

## Question

The broad non-modular benchmark uses values `0..7`, but the largest valid
target in the training distribution is `7 * 7 * 7 + 64 = 407`. The baseline
uses a 512-class output head, leaving a substantial unused tail. This control
keeps the typed-write architecture unchanged and reduces the output head to
448 classes, which still covers every possible target in the benchmark.

This checks whether the broad-value regression is primarily an output-class
optimization problem rather than an internal numeric-state problem.

## Reproduction

Base commit: `10eed11` (`reject operation transition ablation`).

The first draft used 192 classes, but that was invalid because the training
set includes `multiply -> multiply`, whose maximum target is 407 after the
offset. That draft was not measured; the valid run below uses 448 classes.

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_broad_values_compact_head.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_broad_values_compact_head_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_broad_values_compact_head_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_broad_values_compact_head.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_broad_values_compact_head_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_broad_values_compact_head_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|
| 17 | 68.3594% | 71.1914% | 65.5273% | 3,470,659 | 1,668,944 |
| 18 | 68.8477% | 73.3398% | 64.3555% | 3,470,659 | 1,668,944 |
| **mean** | **68.6035%** | 72.2656% | 64.9414% | — | — |

The compact head is 0.5615 points below the V0.69 typed-write 3,000-step
baseline of 69.1650%. It also lowers the total parameter count by 24,640 and
the active estimate by 24,640, but this efficiency change does not improve
quality.

## Decision

Rejected as an architecture fix. The unused output tail is not the main cause
of the broad-value gap. The remaining problem is inside the learned
value/composition interface, not simply the number of output logits.

The 192-class draft failed before training with an expected class-range assert
and was corrected before any result was recorded. The valid experiment passed
the full test suite: `82 passed, 2 warnings` (the existing Transformer
nested-tensor warnings).
