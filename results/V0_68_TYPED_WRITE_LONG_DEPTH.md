# V0.68 Typed Write on Long Held-Out Depths

## Question

V0.67 showed that an operation-typed write adapter improves prior-free,
non-modular two-operation composition. This follow-up tests whether the same
interface remains stable when the recurrent program is trained at depths 1--4
and evaluated at unseen depths 5--8.

## Configuration

- Model: `ne_dynamic_300m_depth8_typed_write_adapter`
- `max_ops: 8`, `train_max_ops: 4`, `seq_len: 18`
- Mod-64 dynamic composition benchmark; train and eval values 0--63
- Operation routing adapter: rank 16, scale 1.0
- Typed write adapter: rank 16, scale 1.0
- Circuit residual: 1.0; Macro-Cells: disabled
- Factorized bank: 23,600 virtual circuits, 154 factor rows, top-8 active
- Steps: 3,000 screen and 9,000 full run; batch size: 512
- Evaluation: 128 examples per held-out depth (5, 6, 7, 8)
- Device: NVIDIA GeForce RTX 3060, CUDA
- Source state: `cf4fa5c`

The 9,000-step commands were:

```text
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_depth8_typed_write_adapter.yaml --steps 9000 --seed 17 --run-id ne_dynamic_300m_depth8_typed_write_adapter_seed17_9000 --checkpoint results/checkpoints/ne_dynamic_300m_depth8_typed_write_adapter_seed17_9000.pt --heldout-depths --examples-per-depth 128 --log-every 1000

python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_depth8_typed_write_adapter.yaml --steps 9000 --seed 18 --run-id ne_dynamic_300m_depth8_typed_write_adapter_seed18_9000 --checkpoint results/checkpoints/ne_dynamic_300m_depth8_typed_write_adapter_seed18_9000.pt --heldout-depths --examples-per-depth 128 --log-every 1000
```

## Results

### 3,000-step screen

| seed | train | held-out depths 5--8 | depth 5 | depth 6 | depth 7 | depth 8 |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 98.24% | 96.29% | 98.44% | 98.44% | 92.19% | 96.09% |
| 18 | 97.85% | 93.75% | 96.09% | 92.19% | 92.97% | 93.75% |
| mean | 98.05% | **95.02%** | 97.27% | 95.31% | 92.58% | 94.92% |

### 9,000-step validation

| seed | train | held-out depths 5--8 | depth 5 | depth 6 | depth 7 | depth 8 |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.00% | **99.41%** | 100.00% | 99.22% | 98.44% | 100.00% |
| 18 | 100.00% | **99.41%** | 99.22% | 99.22% | 100.00% | 99.22% |
| mean | 100.00% | **99.41%** | 99.61% | 99.22% | 99.22% | 99.61% |

The model has 3,336,643 stored parameters and an estimated 1,534,928 active
parameters (46.00% active fraction). At 9,000 steps, seed 17 routed through
153/154 factor rows and seed 18 through 151/154; virtual-bank utilization was
18.67% and 16.61%, respectively.

## Decision

The typed write interface generalizes cleanly to unseen recurrent depths. It
reaches 99.41% mean at 9,000 steps, essentially matching the earlier 20M-class
long-depth reference (99.44%) while using the same sparse active budget. The
3,000-step screen is substantially stronger than the earlier plain 300M
depth-8 screen (75.68% at seed 17), showing faster optimization rather than a
mere late-training effect.

This validates typed-write as the current interface reference, but it does not
demonstrate a final capacity gain: the long-depth task is already near
saturation, and the 300M virtual bank still activates only about 1.53M
parameters. The next meaningful test is broader non-modular values and longer
operation compositions, followed by capacity scaling only if that benchmark
still has headroom.

JSON evidence is stored in:

- `results/runs/ne_dynamic_300m_depth8_typed_write_adapter_seed17_3000.json`
- `results/runs/ne_dynamic_300m_depth8_typed_write_adapter_seed18_3000.json`
- `results/runs/ne_dynamic_300m_depth8_typed_write_adapter_seed17_9000.json`
- `results/runs/ne_dynamic_300m_depth8_typed_write_adapter_seed18_9000.json`
