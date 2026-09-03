# V0.69 Broader Non-Modular Value Interface

## Question

V0.67's typed-write adapter reaches 89.27% mean on prior-free non-modular
composition when operands are limited to 0--3. This test expands the operand
domain to 0--7 and the output head to 512 classes. The goal is to determine
whether the gain reflects a compositional interface or only a narrow small-
value regime.

## Configuration

- Base: `ne_dynamic_300m_composition_typed_write_adapter`
- Operand values: 0--7 for both train and evaluation
- Arithmetic: ordinary integer arithmetic (`generator_modulus: null`)
- Target offset: 64; output classes: 512
- Held-out pairs: `add -> multiply`, `multiply -> add`
- Operation routing adapter: rank 16; typed write adapter: rank 16
- Circuit residual: 1.0; Macro-Cells: disabled
- Steps: 3,000 screen and 9,000 full runs; batch size: 512
- Device: NVIDIA GeForce RTX 3060, CUDA
- Source state: `741f637`

The 9,000-step learned-encoder commands were:

```text
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_broad_values.yaml --steps 9000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_broad_values_seed17_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_broad_values_seed17_9000.pt --examples-per-task 1024 --log-every 1000

python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_broad_values.yaml --steps 9000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_broad_values_seed18_9000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_broad_values_seed18_9000.pt --examples-per-task 1024 --log-every 1000
```

## Results

### Learned value encoder

| steps | seed | train | held-out | add -> multiply | multiply -> add |
|---:|---:|---:|---:|---:|---:|
| 3,000 | 17 | 100.00% | 67.87% | 70.80% | 64.94% |
| 3,000 | 18 | 100.00% | 70.46% | 77.34% | 63.57% |
| 3,000 mean | -- | 100.00% | 69.17% | -- | -- |
| 9,000 | 17 | 100.00% | 74.80% | 81.74% | 67.87% |
| 9,000 | 18 | 100.00% | 71.88% | 77.73% | 66.02% |
| 9,000 mean | -- | 100.00% | **73.34%** | 79.74% | 66.94% |

The broad-value model has 3,495,299 stored parameters and an estimated
1,693,584 active parameters (48.45% active fraction). The 9,000-step mean is
15.93 points below V0.67's 89.27% mean on values 0--3.

### Hybrid Fourier screen

As a focused representation ablation, `value_encoder_mode: hybrid_fourier` was
also screened for 3,000 steps:

| seed | held-out | add -> multiply | multiply -> add |
|---:|---:|---:|---:|
| 17 | 70.56% | 75.20% | 65.92% |
| 18 | 67.29% | 71.29% | 63.28% |
| mean | **68.92%** | 73.24% | 64.60% |

The hybrid screen does not provide a meaningful improvement over the learned
encoder screen and was not extended to 9,000 steps.

## Decision

The typed-write gain is real on the narrow 0--3 non-modular benchmark, but it
does not transfer automatically to a wider value domain. More virtual
circuits or a larger active path are not justified by this result: training
accuracy is already 100%, while held-out composition remains low.

This experiment is **rejected as a broad-domain solution**. Hybrid Fourier is
also rejected as the immediate fix. The evidence localizes the remaining
bottleneck to numeric state representation/output interface and composition
transfer, not raw routing capacity. The next architecture should expose a
small structured numeric working state or equivalent compositional interface,
then be re-tested before any 500M/700M/1B scaling.

JSON evidence is stored in `results/runs/` under the corresponding broad-value
and hybrid-broad-value run IDs.
