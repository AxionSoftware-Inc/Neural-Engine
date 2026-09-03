# V0.71 Operation-Specific Transition Ablation

## Question

The typed-write adapter improved the narrow prior-free composition task, but
the broad non-modular value screen remained the main bottleneck. This ablation
tests whether adding a second operation-conditioned low-rank nonlinear
transition before the shared register writer improves numeric composition
transfer without changing the sparse circuit budget.

The transition is applied to `query + delta` immediately before the shared
writer. It has one rank-16 down/up pair and bias per operation, with residual
scale `1.0`. The existing operation adapter, typed-write adapter, sparse
factorized circuit bank, and broad-value benchmark are otherwise unchanged.

## Reproduction

Base commit before the ablation: `41e1b4b` (`test structured numeric state`).

```powershell
$env:PYTHONPATH='.'
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_transition_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_transition_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_transition_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_transition_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_transition_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_transition_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|
| 17 | 64.9414% | 68.2617% | 61.6211% | 3,533,315 | 1,731,600 |
| 18 | 69.0430% | 73.1445% | 64.9414% | 3,533,315 | 1,731,600 |
| **mean** | **66.9922%** | 70.7031% | 63.2813% | — | — |

The two-seed screen is 2.1729 points below the V0.69 typed-write baseline
(69.1650%) and 3.3203 points below the V0.70 16D numeric-state screen
(70.3125%). The operation transition also adds parameters and active work,
but provides no quality gain.

## Decision

Rejected as the default architecture. The extra low-rank transform appears to
interfere with or duplicate the already operation-conditioned write interface
instead of repairing the numeric state representation. This is evidence that
adding another generic operation-conditioned layer is not the missing
fundamental mechanism; it should not be repeated separately for every larger
model size.

The current learned reference remains the V0.67 typed-write adapter. For the
broad 0--7 non-modular task, the best tested short screen is the V0.70 16D
structured numeric state at 70.31%, while V0.69's longer 9,000-step typed-write
run reaches 73.34%. The next architecture work should target a genuinely
structured value/composition interface rather than stacking another generic
transition.

After the source change, the full test suite passed: `82 passed, 2 warnings`
(the warnings are the existing Transformer nested-tensor warnings).
