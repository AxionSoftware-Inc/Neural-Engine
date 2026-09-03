# V0.67 Operation-Typed Write Adapter

## Motivation

The operation-conditioned adapter in V0.63 changes the routing query, but the
final accumulator update is still produced by one shared `register_writer`.
That shared writer may mix the transition interfaces for addition,
subtraction, and multiplication. This experiment adds a small per-operation
low-rank adapter after the shared writer, so the write itself is typed while
the sparse circuit bank and routing budget remain unchanged.

## Configuration

- Base: `ne_dynamic_300m_composition_nonmodular_operation_adapter`
- New path: `operation_write_adapter_rank: 16`, scale 1.0
- Existing routing adapter: rank 16, scale 1.0, fixed (ungated)
- Circuit residual: 1.0
- Modular prior: disabled; Macro-Cells: disabled
- Held-out pairs: `add -> multiply`, `multiply -> add`
- Train/eval values: 0--3, ordinary integer arithmetic, target offset 16
- Steps: 9,000; batch size: 512; examples per task: 1,024
- Device: NVIDIA GeForce RTX 3060, CUDA
- Source state before the experiment: `241b120`

Commands:

```text
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter.yaml --steps 9000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_seed17_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_seed17_9000.pt --examples-per-task 1024 --log-every 1000

python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter.yaml --steps 9000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_seed18_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_seed18_9000.pt --examples-per-task 1024 --log-every 1000

python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter.yaml --steps 9000 --seed 19 --run-id ne_dynamic_300m_composition_typed_write_adapter_seed19_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_seed19_9000.pt --examples-per-task 1024 --log-every 1000
```

## Results

| seed | train | held-out | add -> multiply | multiply -> add |
|---:|---:|---:|---:|---:|
| 17 | 100.00% | 92.72% | 90.82% | 94.63% |
| 18 | 100.00% | 84.67% | 81.45% | 87.89% |
| 19 | 100.00% | 90.43% | 87.30% | 93.55% |
| mean | 100.00% | **89.27%** | 86.52% | 92.02% |

The model has 3,322,819 stored parameters and an estimated 1,521,104 active
parameters (45.78% active fraction). The full operation-adapter reference
from V0.63 is 84.18% mean across two seeds; the typed-write adapter improves
the mean by 5.09 percentage points. The no-residual control from V0.65 is
79.20% mean.

## Decision

This is the strongest learned, prior-free, non-modular composition result so
far. The gain survives three seeds and uses the same sparse routing capacity;
it does not come from activating a larger fraction of the virtual bank. The
evidence supports accepting the typed write path as the current architecture
reference.

The result is not yet a proof of billion-parameter scaling. Seed variance
remains material, and the benchmark has only two held-out operation orders
with values 0--3. The next validation should test the same typed interface on
longer compositions and broader operand ranges before scaling the bank.

JSON evidence is stored in:

- `results/runs/ne_dynamic_300m_composition_typed_write_adapter_seed17_9000.json`
- `results/runs/ne_dynamic_300m_composition_typed_write_adapter_seed18_9000.json`
- `results/runs/ne_dynamic_300m_composition_typed_write_adapter_seed19_9000.json`
