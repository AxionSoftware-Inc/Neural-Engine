# NE-V0.12 numeric encoding plus adaptive execution

V0.12 combines the two strongest ideas found so far:

- structured 64-modular numeric/Fourier operand encoding from V0.9;
- learned adaptive halting with final-target supervision at the selected exit
  step from V0.11.

The core remains attention-free: a slot encoder, persistent recurrent state,
hierarchical router, and low-rank micro-circuit bank.

## Full balanced benchmark

Hardware: NVIDIA GeForce RTX 3060, `torch 2.6.0+cu124`, seed 17, balanced task
batches, 5,000 steps, batch size 128.

| Metric | NE-V0.9 fixed | NE-V0.12 adaptive |
|---|---:|---:|
| Total parameters | 20.246M | 20.247M |
| Estimated unique active fraction | 9.77% | 9.77% |
| Exact accuracy | 71.59% | **71.98%** |
| Depth-1 accuracy | 94.92% | 94.75% |
| Depth-2 accuracy | 41.67% | **43.10%** |
| Depth-3 accuracy | 31.51% | **32.55%** |
| Training time | **165.5 s** | 170.9 s |
| Training throughput | **3,867/s** | 3,745/s |
| Average executed steps | 3.00 | **1.60** |
| Active-step fraction | 100% | **53.2%** |

The learned schedule is task-aligned: depth-1, depth-2, and depth-3 tasks
execute 1.00, 2.00, and 2.99 steps on average.

## Held-out combination benchmark

All operand values are visible during training; 25% of task-aware operand
combinations are held out.

| Model | Train split | Held-out combinations | Average steps |
|---|---:|---:|---:|
| NE-V0.3 fixed lookup | 68.98% | 49.95% | 3.00 |
| Transformer | 51.02% | 50.81% | dense |
| NE-V0.9 fixed numeric | **71.04%** | **57.11%** | 3.00 |
| NE-V0.12 adaptive numeric | 71.41% | 55.68% | **1.60** |

## Decision

**Recommend V0.12 when active-compute efficiency matters.** It is the highest
full-benchmark score so far and cuts recurrent routed execution by about 46.7%
while preserving most held-out generalization. Keep V0.9 as the maximum-quality
held-out reference until adaptive exit training closes the remaining 1.43-point
held-out gap.

The current step fraction is measured from trained validation runs. Wall-clock
inference needs checkpoint-backed benchmarking because dynamic sub-batching can
add GPU indexing overhead.
