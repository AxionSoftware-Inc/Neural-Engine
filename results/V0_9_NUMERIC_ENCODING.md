# NE-V0.9 structured numeric value encoding

V0.9 replaces learned lookup embeddings for operand values with a shared
64-modular Fourier/numeric encoder. Task identity remains learned, while value
tokens are represented by a normalized scalar and harmonics of the modular
value. This removes the need to learn a separate arbitrary vector for every
operand value and keeps the recurrent routed core unchanged.

## Original balanced benchmark

Hardware: NVIDIA GeForce RTX 3060, `torch 2.6.0+cu124`, seed 17, balanced task
batches, 5,000 steps, batch size 128.

| Metric | NE-V0.3 lookup values | NE-V0.9 numeric values | Transformer |
|---|---:|---:|---:|
| Total parameters | 20.284M | 20.246M | 20.582M |
| Estimated active fraction | 9.93% | 9.77% | 100% |
| Exact accuracy | 68.59% | **71.59%** | 51.02% |
| Depth-1 accuracy | 97.40% | 94.92% | 70.88% |
| Depth-2 accuracy | 37.76% | **41.67%** | 35.42% |
| Depth-3 accuracy | 13.02% | **31.51%** | 7.03% |
| Training time | **135.2 s** | 165.5 s | 543.3 s |
| Training throughput | **4,733/s** | 3,867/s | 1,178/s |

## Held-out combination benchmark

All values `0..63` are visible in training. A stable task-aware hash withholds
25% of operand combinations from validation.

| Model | Train split | Held-out combinations |
|---|---:|---:|
| NE-V0.3 | 68.98% | 49.95% |
| Transformer | 51.02% | 50.81% |
| NE-V0.9 | **71.04%** | **57.11%** |

V0.9 improves held-out depth-3 accuracy from 7.03% (Transformer) and 10.94%
(V0.3 in this split) to 27.99%. It still does not solve the disjoint-value
stress test, where the result is 14.14%; that result remains documented in
`results/HELDOUT_COMPOSITION.md`.

## Decision

**Promote V0.9 as the new quality/reference variant.** Structured value
encoding improves both ordinary accuracy and unseen-combination
generalization while preserving the attention-free sparse architecture. Its
current cost is approximately 22% lower training throughput than V0.3 due to
the additional feature construction; inference and active circuit budget
remain structurally sparse.
