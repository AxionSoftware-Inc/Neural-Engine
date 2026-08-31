# NE-V0.8 serial circuit composition ablation

V0.8 keeps the V0.3 slot encoder, router, circuit bank size, and three-step
recurrent loop. The selected eight micro-circuits are applied sequentially;
each circuit receives the residual produced by the previous one. The default
V0.3 mode mixes the same eight selected circuits in parallel.

## Controlled CUDA result

Hardware: NVIDIA GeForce RTX 3060, `torch 2.6.0+cu124`, seed 17, balanced
training batches, 5,000 steps, batch size 128.

| Metric | NE-V0.3 parallel | NE-V0.8 serial |
|---|---:|---:|
| Total parameters | 20.284M | 20.284M |
| Estimated active parameters | 2.015M | 2.015M |
| Estimated active fraction | 9.93% | 9.93% |
| Exact accuracy | **68.59%** | 68.44% |
| Depth-1 accuracy | 97.40% | 97.74% |
| Depth-2 accuracy | 37.76% | **38.54%** |
| Depth-3 accuracy | **13.02%** | 10.42% |
| Training time | **135.2 s** | 277.9 s |
| Training throughput | **4,733/s** | 2,303/s |
| Circuits used | 1,395 / 1,408 | 1,388 / 1,408 |

## Decision

**Reject V0.8 as the default.** Serial composition slightly helps depth-2 but
does not solve depth-3 and costs about 2.1x training time on this GPU. Keep the
serial implementation as an optional research mode because it tests a genuine
composition mechanism under the same unique active-circuit budget.
