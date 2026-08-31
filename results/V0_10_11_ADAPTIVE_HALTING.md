# NE-V0.10/V0.11 adaptive halting

V0.10 adds a halt head after every recurrent step. During training, the halt
head receives the known synthetic task depth as a learning signal; during
inference, it decides whether to execute another routed circuit step. V0.11
also trains the output at the predicted exit step against the final target.

No attention or Transformer block is added. The maximum execution path remains
three recurrent steps with eight active circuits per step.

## Controlled CUDA result

Hardware: NVIDIA GeForce RTX 3060, `torch 2.6.0+cu124`, seed 17, balanced
training batches, 5,000 steps, batch size 128.

| Metric | NE-V0.3 fixed 3 steps | V0.10 adaptive | V0.11 adaptive + exit loss |
|---|---:|---:|---:|
| Total parameters | 20.284M | 20.285M | 20.285M |
| Exact accuracy | **68.59%** | 64.77% | 68.02% |
| Depth-1 accuracy | 97.40% | 92.01% | 97.61% |
| Depth-2 accuracy | **37.76%** | 36.59% | 36.46% |
| Depth-3 accuracy | **13.02%** | 11.20% | 10.81% |
| Average executed steps | 3.00 | 1.60 | 1.60 |
| Active-step fraction | 100% | 53.3% | 53.3% |
| Training time | 135.2 s | 159.0 s | 161.9 s |

V0.11 learned the intended schedule: depth-1 tasks execute 1.00 steps on
average, depth-2 tasks 2.00, and depth-3 tasks 2.99.

## Decision

Keep V0.11 as an optional fast-inference mode, not the default quality mode. It
loses 0.57 percentage points of balanced accuracy while cutting recurrent
routed execution by about 46.7%. The default remains V0.3 fixed three-step
execution until a learned halt policy reaches parity on depth-2/3 tasks.
