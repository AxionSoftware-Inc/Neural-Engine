# NE-V0.4 five-step ablation

V0.4 keeps the V0.3 slot encoder and one-address router but increases recurrent
internal steps from 3 to 5. The purpose was to test whether additional
reasoning depth alone improves composition.

## Result

| Metric | NE-V0.3, 3 steps | NE-V0.4, 5 steps |
|---|---:|---:|
| Total parameters | 20.28M | 20.28M |
| Estimated active parameters | 2.015M | 2.016M |
| Active circuit blocks per step | 8 | 8 |
| Balanced exact accuracy | **68.59%** | 66.41% |
| Validation loss | **1.0579** | 1.1059 |
| Depth-1 accuracy | **97.40%** | 95.14% |
| Depth-2 accuracy | **37.76%** | 37.63% |
| Depth-3 accuracy | **13.02%** | 8.98% |
| Training time | **135.2 s** | 182.2 s |
| Training throughput | **4,733/s** | 3,512/s |
| Dead circuit fraction | 0.92% | **0.36%** |

## Interpretation

More recurrent steps improve route coverage slightly but do not improve
multi-step task accuracy. They reduce overall quality and cost about 35% more
training time in this implementation. The recurrent state needs better
composition/supervision, not simply a longer unrolled path.

Conclusion: **REJECT as default; retain 3 internal steps for V0.3.**

Next work will target circuit specialization, intermediate computation signals,
and held-out compositions while keeping the active circuit budget fixed.
