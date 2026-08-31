# NE-V0.5 intermediate-supervision ablation

V0.5 keeps the best V0.3 architecture—slot encoder, one-address hierarchical
router, 8 active circuits, and 3 recurrent steps—and adds optional supervision
for deterministic partial results at each recurrent step. The model remains
attention-free and contains no Transformer blocks.

## Controlled CUDA result

Hardware: NVIDIA GeForce RTX 3060, `torch 2.6.0+cu124`, seed 17, balanced
training batches, 5,000 steps, batch size 128.

| Metric | NE-V0.3 | NE-V0.5 |
|---|---:|---:|
| Total parameters | 20.284M | 20.284M |
| Estimated active parameters | 2.015M | 2.015M |
| Estimated active fraction | 9.93% | 9.93% |
| Exact accuracy | 68.59% | 68.18% |
| Depth 1 accuracy | 97.40% | 97.27% |
| Depth 2 accuracy | 37.76% | 37.11% |
| Depth 3 accuracy | 13.02% | 11.98% |
| Training time | 135.2 s | 149.0 s |
| Training throughput | 4,733 samples/s | 4,295 samples/s |
| Circuits used | 1,395 / 1,408 | 1,393 / 1,408 |
| Dead circuit fraction | 0.92% | 1.07% |

## Decision

**Reject V0.5 as the default.** The auxiliary targets increased training cost
and did not improve final accuracy or multi-step composition. V0.3 remains the
main architecture and the reference point for further experiments.

This is a controlled ablation, not evidence that intermediate supervision is
generally ineffective. The current synthetic task decomposition may not match
the state representation learned by the router and circuits.
