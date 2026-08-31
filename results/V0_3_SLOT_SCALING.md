# NE-V0.3 slot encoder and fixed-active scaling

V0.3 replaces mean-only input pooling with a small attention-free structured
slot encoder. The first five input slots are position-conditioned, flattened,
and projected into the recurrent state. This preserves operand order while
keeping the core computation recurrent + routed micro-circuits.

## Main V0.3 comparison

| Metric | NE-20 V0.3 | Dense Transformer |
|---|---:|---:|
| Total parameters | 20.28M | 20.58M |
| Estimated active parameters | 2.015M | 20.58M |
| Active fraction | 9.93% | 100% |
| Balanced exact accuracy | **68.59%** | 51.02% |
| Depth-1 accuracy | **97.40%** | 70.88% |
| Depth-2 accuracy | **37.76%** | 35.42% |
| Depth-3 accuracy | **13.02%** | 7.03% |
| Training time | **135.2 s** | 543.3 s |
| Training throughput | **4,733/s** | 1,178/s |
| Inference throughput | **21,324/s** | 3,721/s |

The models are parameter-matched within 1.5%. NE-20 V0.3 uses approximately
10.2x fewer active parameters than the dense baseline while achieving higher
balanced accuracy in the same 5,000-step experiment.

## Fixed-active capacity scaling

All three NE models use 8 active circuits per internal step and 3 recurrent
steps. Only the stored circuit bank grows.

| Model | Total params | Active params | Active fraction | Accuracy | Depth-1 | Depth-2 | Depth-3 | Circuits used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NE-20 V0.3 | 20.28M | 2.015M | 9.93% | 68.59% | 97.40% | 37.76% | 13.02% | 1,395 / 1,408 |
| NE-50 V0.3 | 50.37M | 2.015M | 4.00% | 67.71% | 97.31% | 36.33% | 10.29% | 3,229 / 3,712 |
| NE-100 V0.3 | 100.50M | 2.018M | 2.01% | 68.41% | 97.87% | 36.07% | 12.37% | 7,003 / 7,552 |

Training times were 135.2s, 194.4s, and 302.8s respectively. Inference
throughput was 21,324/s, 17,876/s, and 18,841/s respectively on the RTX 3060.

## Interpretation

The architecture passes the fixed-active test: total capacity grows from
20.28M to 100.50M (4.95x) while estimated active parameters remain within
0.15% of 2.01M and quality stays within 0.88 percentage points. This is strong
evidence that stored capacity can be increased without proportional inference
compute growth.

The expected quality scaling did not appear: NE-50 and NE-100 do not beat
NE-20. Therefore dormant capacity is currently not being converted into new
useful skills. The next bottleneck is circuit specialization/composition, not
the ability to store more circuits.

## Routing health

- NE-20 dead circuits: `0.92%`
- NE-50 dead circuits: `13.01%`
- NE-100 dead circuits: `7.27%`
- NE-100 router depth: 5, enough for all 7,552 circuits (`8^5 = 32,768`)
- No collapse onto a tiny fixed circuit subset was observed.

## Decision

**Retain V0.3 as the main architecture.** It is the best result so far:
parameter-matched quality is substantially above the baseline, active compute is
about one tenth of dense, and the fixed-active property survives 100M total
capacity. Next work should focus on learned circuit composition, held-out
multi-hop tasks, and exact active byte/FLOP accounting rather than increasing
model size again.
