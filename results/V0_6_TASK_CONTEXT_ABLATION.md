# NE-V0.6/V0.7 task-context ablation

These variants keep the V0.3 slot encoder, one-address hierarchical router,
three recurrent steps, and eight active circuits. They add a small direct task
identity embedding so the router can condition each step on the operator. No
attention or Transformer block is introduced.

Hardware: NVIDIA GeForce RTX 3060, `torch 2.6.0+cu124`, seed 17, balanced
training batches, 5,000 steps, batch size 128.

| Metric | NE-V0.3 | NE-V0.6: query + update | NE-V0.7: query only |
|---|---:|---:|---:|
| Total parameters | 20.284M | 20.290M | 20.290M |
| Estimated active parameters | 2.015M | 2.021M | 2.021M |
| Estimated active fraction | 9.93% | 9.96% | 9.96% |
| Exact accuracy | **68.59%** | **68.72%** | 68.12% |
| Depth-1 accuracy | 97.40% | **98.35%** | 97.53% |
| Depth-2 accuracy | **37.76%** | 36.85% | 36.85% |
| Depth-3 accuracy | **13.02%** | 11.72% | 11.20% |
| Training time | 135.2 s | **133.6 s** | 138.5 s |
| Circuits used | 1,395 / 1,408 | 1,382 / 1,408 | 1,378 / 1,408 |

## Decision

**Reject V0.6/V0.7 as the default.** The direct task signal slightly raises
overall accuracy in one seed/configuration, but it hurts depth-2/3 composition
and adds parameters without improving the main bottleneck. V0.3 remains the
cleaner and stronger reference architecture.
