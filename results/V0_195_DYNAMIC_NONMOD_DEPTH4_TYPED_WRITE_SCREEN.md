# V0.195 Non-Modular Depth-4 Typed-Write Screen

## Question

V0.67 showed strong prior-free composition on two operations with values
0--3. V0.68 showed that typed-write generalizes to deeper programs in the
mod-64 setting. This screen combines the harder parts: ordinary integer
arithmetic (`modulus: null`) and unseen program depths. Training exposes only
depths 1--2; evaluation uses unseen depths 3--4.

## Configuration

- architecture: `DynamicRegisterNeuralEngine`, attention-free;
- 300M-class factorized virtual bank: 23,600 virtual circuits and 154 factor
  rows;
- operation-specific circuit banks, rank-16 operation query adapter, and
  rank-16 typed-write adapter;
- active route: 8 circuits per executed step, about 2.10M estimated active
  parameters in this four-step configuration;
- values 0--3, ordinary add/subtract/multiply, target offset 64, 512 output
  classes;
- batch size 512, 3,000 steps, stage-loss weight 0.5, CUDA on RTX 3060.

The target offset is only a label-layout detail: it keeps negative non-modular
results inside the classifier range and does not provide an arithmetic prior.

## Results

| seed | train depth 1--2 | held-out depth 3--4 | depth 3 | depth 4 |
|---:|---:|---:|---:|---:|
| 17 | 100.00% | 83.20% | 91.41% | 75.00% |
| 18 | 100.00% | 87.11% | 92.19% | 82.03% |
| **mean** | **100.00%** | **85.16%** | **91.80%** | **78.52%** |

Both runs converged to near-zero training loss. Held-out quality is positive
on both seeds, but depth-4 remains the bottleneck and the screen is not yet a
scaling claim. The 3k budget is also shorter than the 9k validation used by
V0.68.

## Decision

**POSITIVE SCREEN — VALIDATION REQUIRED.** This is the first combined
non-modular/depth-generalization signal for the typed-write DynamicRegister
track. It supports testing the same frozen architecture for 9,000 steps and
then adding a third seed if the gain remains stable. Do not increase the
virtual bank to 500M/700M/1B from this screen alone.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter.yaml`
- `results/runs/nonmod_depth4_typed_write_seed17_3000.json`
- `results/runs/nonmod_depth4_typed_write_seed18_3000.json`
- `neural_engine/dynamic_register.py` (optional non-modular modulus support)
- `data/dynamic_composition.py` (offset-aware non-modular targets)
