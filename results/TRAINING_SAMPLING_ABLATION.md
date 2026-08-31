# Composition-focused training sampling ablation

This experiment changes only the training distribution. Evaluation remains
uniformly balanced across all 15 task IDs. Depth-1, depth-2, and depth-3 task
weights are `1`, `1 + strength`, and `1 + 2*strength` respectively.

Hardware: NVIDIA GeForce RTX 3060, `torch 2.6.0+cu124`, seed 17, NE-V0.3,
5,000 steps, batch size 128.

| Training mixture | Overall | Depth 1 | Depth 2 | Depth 3 | Training time |
|---|---:|---:|---:|---:|---:|
| Uniform task-balanced | **68.59%** | **97.40%** | **37.76%** | 13.02% | 135.2 s |
| Strength 0.5 (`1:1.5:2`) | 63.70% | 89.28% | 35.94% | **14.71%** | 135.3 s |
| Strength 1.0 (`1:2:3`) | 62.42% | 87.02% | 35.68% | **15.36%** | 137.6 s |

## Decision

Composition-focused sampling improves depth-3 accuracy by 1.69–2.34
percentage points, but it reduces overall balanced accuracy because the easy
depth-1 task families receive fewer updates. Keep uniform task-balanced
training as the default; expose composition sampling as an optional recipe for
users specifically optimizing multi-step tasks.
