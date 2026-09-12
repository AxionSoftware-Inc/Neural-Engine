# V0.302 — Operator-Valued Product Encoder Audit

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Question

The prior-free typed-carry model uses a learned dense transform on the
elementwise product `accumulator * operand` before routing. V0.302 replaces
that transform with a shared operator-valued linear map: packet width `16`,
eight learned operator-basis matrices, and coefficient mixing across packets.
The hypothesis was that reusable operator structure could lower active cost
without losing the product signal needed by the sparse engine.

The rest of the model is the V0.298 numeric-convolution configuration:

- 300M virtual factorized circuit bank;
- base-16 typed carry state and learned numeric multiply convolution;
- train values `0..95`, train depths `1..2`, held-out depths `3..4`;
- 2,000 steps, batch size 128, 1,024 examples per depth;
- seeds 17 and 18;
- matched dense-product control retrained with the same commands.

## Parameter and end-to-end result

| Metric | Dense control mean | Operator-valued mean | Delta |
|---|---:|---:|---:|
| total parameters | 7,350,295 | 7,209,495 | −140,800 |
| active-path estimate | 2,051,136 | 1,910,336 | −140,800 |
| held-out accuracy | 4.1504% | 8.3008% | +4.1504 pp |
| held-out depth-3 accuracy | 5.3223% | 11.6211% | +6.2988 pp |
| held-out depth-4 accuracy | 2.9785% | 4.9805% | +2.0020 pp |
| mean held-out CE | 12.0393 | 12.5253 | +0.4860 |

The seed-specific held-out accuracy was `3.3691% → 4.7852%` for seed17 and
`4.9316% → 11.8164%` for seed18. The hard-accuracy improvement is therefore
seed-sensitive and is accompanied by worse mean CE.

## Fixed operation-wise result

Each cell uses 512 deterministic homogeneous programs and is evaluated on the
same checkpoint pair.

| Operation | Dense d3 | Operator d3 | Delta | Dense d4 | Operator d4 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| add | 11.3281% | 30.2734% | +18.9453 pp | 2.9297% | 10.9375% | +8.0078 pp |
| subtract | 8.7891% | 28.1250% | +19.3359 pp | 3.8086% | 8.9844% | +5.1758 pp |
| multiply | 3.4180% | 3.3203% | −0.0977 pp | 3.1250% | 3.0273% | −0.0977 pp |

The main multiply bottleneck is therefore unchanged. The apparent aggregate
gain is an additive/subtractive operation effect, not evidence that the
operator-valued product map learned multiplication.

## Decision

- **Retain as opt-in efficiency/quality diagnostic.** It saves 140,800 stored
  and estimated active parameters and improves add/subtract on this screen.
- **Do not make it the default and do not scale it to 700M/1B as a multiply
  solution.** CE is worse and multiply is flat/slightly negative.
- The remaining hypothesis is that a single product transform is shared across
  incompatible operation semantics. The next controlled architecture test is
  shared operator basis with operation-conditioned coefficient maps, with a
  matched dense control and the same fixed multiply screen.

## Artifacts

- Config:
  `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_operator_valued_product.yaml`
- Operator seed17 operation-wise result:
  `results/operationwise_fixed_checkpoint_eval_v0_302_seed17.json`
- Operator seed18 operation-wise result:
  `results/operationwise_fixed_checkpoint_eval_v0_302_seed18.json`
- Dense seed17 operation-wise result:
  `results/operationwise_fixed_checkpoint_eval_v0_302_dense_control_seed17.json`
- Dense seed18 operation-wise result:
  `results/operationwise_fixed_checkpoint_eval_v0_302_dense_control_seed18.json`
