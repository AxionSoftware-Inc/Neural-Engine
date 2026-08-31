# NE-V0.1 composition comparison

V0.1 adds two attention-free changes to NE-V0:

- position-conditioned token mixing before pooling, so operand order is
  preserved without attention;
- an explicit learned embedding for each recurrent internal step, so routing
  can choose a different computation path at each step.

## Controlled run

- Commit: `90b8dde` plus the V0.1 changes in the next commit
- GPU: NVIDIA GeForce RTX 3060, 12 GB
- Runtime: `torch 2.6.0+cu124`, CUDA 12.4
- Dataset: same 15-task synthetic benchmark
- Sampling: task-balanced
- Seed: 17
- Steps: 5,000
- Batch size: 128

## Comparison

| Metric | NE-V0 | NE-V0.1 | Dense Transformer |
|---|---:|---:|---:|
| Total parameters | 28.02M | 28.05M | 20.58M |
| Estimated active parameters | 1.396M | 1.422M | 20.58M |
| Active fraction | 4.98% | 5.07% | 100% |
| Balanced exact accuracy | 52.71% | **53.75%** | 51.02% |
| Validation loss | 1.5193 | **1.4732** | 1.6479 |
| Depth-1 accuracy | 81.60% | **81.90%** | 70.88% |
| Depth-2 accuracy | 10.29% | **16.15%** | 35.42% |
| Depth-3 accuracy | **8.46%** | 6.90% | 7.03% |
| Training time | 141.3 s | 146.1 s | 543.3 s |
| Training throughput | 4,528.6/s | 4,381.4/s | 1,178.1/s |
| Inference throughput | 19,026/s | **20,163/s** | 3,721/s |

## Important task-level changes

| Task | NE-V0 | NE-V0.1 | Transformer |
|---|---:|---:|---:|
| add | 96.88% | 95.70% | 10.16% |
| subtract | 49.22% | 47.66% | 6.64% |
| multiply | 89.84% | 83.59% | 22.27% |
| lookup | 25.39% | **42.97%** | 100.00% |
| compose_add_mul | 15.63% | 14.84% | 10.55% |
| compose_if | 4.69% | 3.91% | 7.81% |
| state_machine | 5.08% | 1.95% | 2.73% |

## Routing health

- Circuits used: `1817 / 2048`
- Dead circuit fraction: `11.28%`
- Routing entropy: `6.58` nats
- Maximum circuit load: `0.82%`
- Active circuits per internal step: `8`
- Router collapse: not observed

## Interpretation

The V0.1 modifications are retained because they improve total balanced
accuracy, validation loss, and depth-2 performance while preserving the roughly
5% active-parameter budget. The result is still not a solved reasoning model:
depth-2/3 composition remains the primary weakness, and total parameter counts
are not yet exactly matched to the Transformer.

Conclusion: **V0.1 RETAINED — positive active-computation signal, composition still open.**

Next: train an approximately 20M-parameter NE model and a parameter-matched
dense baseline, then run NE-20/NE-50/NE-100 at a fixed active circuit budget.
