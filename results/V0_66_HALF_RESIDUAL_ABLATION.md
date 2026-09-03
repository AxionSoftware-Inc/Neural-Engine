# V0.66 Half-Residual Circuit Ablation

## Question

V0.65 showed that removing the sparse circuit residual hurts the prior-free,
non-modular composition task (79.20% mean across two seeds versus 84.18% with
the full residual). This test checks whether a smaller residual contribution
preserves most of the useful circuit signal while reducing interference with
the operation-conditioned adapter.

## Configuration

- Base: `ne_dynamic_300m_composition_nonmodular_operation_adapter`
- Change: `circuit_residual_scale: 1.0 -> 0.5`
- Operation adapter: rank 16, scale 1.0, fixed (ungated)
- Modular prior: disabled
- Macro-Cells: disabled
- Held-out pairs: `add -> multiply`, `multiply -> add`
- Train/eval values: 0--3, ordinary integer arithmetic, target offset 16
- Steps: 9,000; batch size: 512; examples per task: 1,024
- Device: NVIDIA GeForce RTX 3060, CUDA
- Source state before the experiment: `c3df91e`

Commands:

```text
python -u train_composition.py --config configs/ne_dynamic_300m_composition_nonmodular_operation_adapter_half_residual.yaml --steps 9000 --seed 17 --run-id ne_dynamic_300m_composition_nonmodular_operation_adapter_half_residual_seed17_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_nonmodular_operation_adapter_half_residual_seed17_9000.pt --examples-per-task 1024 --log-every 1000

python -u train_composition.py --config configs/ne_dynamic_300m_composition_nonmodular_operation_adapter_half_residual.yaml --steps 9000 --seed 18 --run-id ne_dynamic_300m_composition_nonmodular_operation_adapter_half_residual_seed18_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_nonmodular_operation_adapter_half_residual_seed18_9000.pt --examples-per-task 1024 --log-every 1000
```

## Results

| seed | train | held-out | add -> multiply | multiply -> add |
|---:|---:|---:|---:|---:|
| 17 | 100.00% | 85.55% | 84.08% | 87.01% |
| 18 | 100.00% | 80.66% | 77.25% | 84.08% |
| mean | 100.00% | **83.11%** | 80.66% | 85.55% |

The model has 3,284,803 stored parameters and an estimated 1,483,088 active
parameters (45.15% active fraction). Both seeds executed the same two recurrent
steps on the held-out examples.

## Decision

The half residual is better than the no-residual control by 3.91 percentage
points, confirming again that the circuit bank supplies useful computation.
However, it is 1.07 points below the full-residual operation-adapter reference
(84.18% mean). Seed variance is also substantial: seed 17 improves over the
full residual, while seed 18 regresses by 3.52 points.

Therefore this ablation is **rejected as the default residual scale**. It is
retained as a documented control, and the current non-modular reference remains
the full operation adapter with `circuit_residual_scale: 1.0`. The result does
not support a claim that tuning residual magnitude solves the architecture
bottleneck or that larger virtual capacity will automatically improve quality.

The JSON evidence is stored in:

- `results/runs/ne_dynamic_300m_composition_nonmodular_operation_adapter_half_residual_seed17_9000.json`
- `results/runs/ne_dynamic_300m_composition_nonmodular_operation_adapter_half_residual_seed18_9000.json`
