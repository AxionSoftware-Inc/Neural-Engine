# Held-out generalization benchmark

The original balanced score is in-distribution: train and validation sample
the same operand distribution. This benchmark adds two stricter splits using
the same 15 synthetic tasks and the same 5,000-step training budget.

Hardware: NVIDIA GeForce RTX 3060, `torch 2.6.0+cu124`, seed 17, batch size
128, balanced task batches.

## 1. Disjoint value range

Training uses operand values `0..31`; validation uses `0..31`; held-out uses
unseen values `32..63`.

| Model | In-range accuracy | Held-out accuracy | Parameters | Active fraction |
|---|---:|---:|---:|---:|
| NE-V0.3 | 84.48% | 9.79% | 20.284M | 9.93% |
| Transformer | 78.26% | 11.20% | 20.582M | 100% |
| NE-V0.9 numeric encoding | 86.46% | **14.14%** | 20.246M | 9.77% |

This is a deliberately harsh distribution-shift test. The numeric encoding
helps, but the low held-out score shows that the routed circuits still learn
value-region-specific behavior.

## 2. Held-out operand combinations

All values `0..63` are visible during training. A stable task-aware hash puts
75% of combinations in the train split and 25% in the held-out split.

| Model | Train-split accuracy | Held-out accuracy | Parameters | Active fraction |
|---|---:|---:|---:|---:|
| NE-V0.3 | **68.98%** | 49.95% | 20.284M | 9.93% |
| Transformer | 51.02% | **50.81%** | 20.582M | 100% |

The held-out gap is much smaller than in the disjoint-value test, proving that
the split is less dominated by unseen token embeddings. However, neither model
shows a convincing compositional advantage yet.

## Decision

Keep NE-V0.3 as the sparse reference model. Do not claim systematic
generalization from the original 68.59% score alone. The next architectural
target is adaptive execution: simple tasks should stop early, while deeper
tasks should receive additional routed computation without activating it for
every input.
