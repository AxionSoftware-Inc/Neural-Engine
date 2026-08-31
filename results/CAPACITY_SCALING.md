# Capacity scaling at fixed active budget

This experiment tests the central scaling hypothesis: increase dormant total
capacity while keeping active computation fixed.

All runs used the same 15-task benchmark, seed 17, balanced task sampling,
batch size 128, 5,000 steps, and 8 active circuits over 3 internal steps.

## Results

| Model | Total params | Active params | Active fraction | Accuracy | Training time | Inference samples/s | Circuits used |
|---|---:|---:|---:|---:|---:|---:|---:|
| NE-20 | 19.69M | 1.422M | 7.22% | 53.62% | 130.9 s | 20,195 | 1,339 / 1,408 |
| NE-50 | 49.77M | 1.422M | 2.86% | **54.77%** | 188.2 s | 20,795 | 2,755 / 3,712 |
| NE-100 | 99.91M | 1.425M | 1.43% | 54.43% | 296.7 s | 19,230 | 6,358 / 7,552 |
| Dense Transformer | 20.58M | 20.58M | 100% | 51.02% | 543.3 s | 3,721 | N/A |

## Interpretation

NE-20 and NE-50 show a positive capacity/compute signal: total capacity grew
2.53x while the active parameter estimate stayed constant, and accuracy rose
from 53.62% to 54.77%. NE-50 also exceeds the parameter-matched dense baseline
by 3.75 percentage points in this run.

The first NE-100 run used a depth-4 router and was invalid for full coverage:
`8^4 = 4096` addresses cannot reach all `7552` circuits. After correcting the
router to depth 5, NE-100 used `6358 / 7552` circuits and reached 54.43%.
It still did not improve over NE-50 after the same 5,000-step budget. The
remaining limiting factor is likely route coverage/load balancing or training
time, not raw stored capacity. This is a meaningful partial result, not
evidence that unlimited dormant capacity is automatically useful.

## Active budget

The active estimate is held constant at approximately 1.422M parameters for all
three NE variants. The circuit-bank portion is exactly 101,376 parameters per
inference step selection (`8` circuits × `3` internal steps); the estimate also
includes the shared encoder, recurrent controller, router projections, local
candidate keys, and output head. It is an analytical model-cost estimate, not
yet a custom-kernel byte counter.

## Decision

**Keep the architecture and focus next on router coverage and composition.**

Next experiments:

1. add router load-balancing and multi-address routing for NE-100;
2. train NE-50/NE-100 longer only after route coverage improves;
3. add held-out composition tasks so the benchmark measures learned algorithms,
   not only interpolation on the existing task families;
4. implement exact active weight-byte/FLOP accounting before any CUDA kernel work.
