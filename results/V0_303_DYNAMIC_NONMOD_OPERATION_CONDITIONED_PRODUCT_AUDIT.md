# V0.303 — Operation-Conditioned Operator Product Audit

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Question

V0.302 showed that a shared operator-valued product transform saved active
parameters and improved add/subtract, but did not improve multiply. V0.303
keeps the shared operator basis but gives add, subtract, and multiply separate
coefficient maps and biases. The hypothesis is that the operation-specific
product semantics were interfering in one shared product map.

The rest of the protocol is the V0.298 prior-free typed-carry/numeric-
convolution model: 300M virtual factorized bank, values `0..95`, train depths
`1..2`, held-out depths `3..4`, base-16 output, and 1,024 examples per depth.
The 2k pilot and 5k extension use matched seeds 17 and 18.

## Parameter budget

The shared basis remains packet width `16` with `8` basis matrices. Only the
packet-mixing coefficients and biases are operation-conditioned.

| Metric | V0.298 reference | V0.303 |
|---|---:|---:|
| total parameters | 7,350,295 | 7,219,479 |
| active-path estimate | 2,051,136 | 1,920,320 |
| product-map scalar DOF | 147,840 dense | 17,024 structured |

Thus the variant does not restore a dense per-operation product matrix.

## End-to-end results

| Steps | Metric | V0.298 mean | V0.303 mean | Delta |
|---:|---|---:|---:|---:|
| 2k | held-out overall | 4.1504%* | 19.6289% | — |
| 2k | held-out depth-3 | — | 25.2441% | — |
| 2k | held-out depth-4 | — | 14.0137% | — |
| 5k | held-out overall | 20.5078% | 24.0723% | +3.5645 pp |
| 5k | held-out depth-3 | 26.6602% | 30.9570% | +4.2969 pp |
| 5k | held-out depth-4 | 14.3555% | 17.1875% | +2.8320 pp |
| 5k | mean held-out CE | 11.7713 | 10.5652 | −1.2061 |

`*` The 2k control number is the matched V0.302 dense-product control, not a
5k V0.298 reference; 5k is the proper quality comparison.

The 5k seed-specific held-out overall accuracy was `24.8047%` (seed17) and
`23.3398%` (seed18). The corresponding depth-4 values were `17.3828%` and
`16.9922%`, so the end-to-end gain is reproducible at both seeds.

## Fixed operation-wise results

Each cell contains 512 deterministic homogeneous programs per seed.

| Operation | V0.303 d3 mean | V0.303 d4 mean | V0.298 d4 mean | V0.303 − V0.298 d4 |
|---|---:|---:|---:|---:|
| add | 77.4414% | 42.9688% | 36.6211% | +6.3477 pp |
| subtract | 83.5938% | 47.3633% | 30.8594% | +16.5039 pp |
| multiply | 3.2227% | 4.8828% | 4.1992% | +0.6836 pp |

The multiply d4 improvement is small and does not generalize to the internal
digits: it is not yet a reliable multiply breakthrough. The large end-to-end
gain is concentrated in add/subtract. At 2k, multiply d4 was `5.6641%` on
average, but after extending to 5k it settled at `4.8828%`; this is another
warning against reading a short pilot as a durable multiply gain.

## High-value stress

The same 5k checkpoints were tested on operands `80..95`, where the previous
V0.299 screen exposed a magnitude regime hidden by ordinary random samples.
Multiply accuracy is `0%` at depths 3 and 4 for both seeds. Add and subtract
also collapse at depth 4 (`0.5859%` and `7.7148%` means). Therefore V0.303
does not solve the high-range value codec/state problem.

## Routing signal

V0.303 increases the number of reached virtual circuits substantially. At 5k,
the training audit reaches roughly `2.4k` unique circuits and the held-out
audit roughly `3.6k`, versus roughly `0.3k–0.5k` in V0.302. This is evidence
that operation-conditioned product features improve useful route separation;
it is not evidence that all `23,600` bank rows are needed or that raw capacity
scaling will fix multiplication.

## Decision

- **Retain V0.303 as the leading opt-in prior-free learned branch.** It gives a
  reproducible `+3.56 pp` 5k held-out gain, `+2.83 pp` depth-4 gain, and a
  lower active budget than V0.298.
- **Do not claim that multiply is solved and do not scale to 700M/1B yet.**
  Fixed multiply is only `+0.68 pp` at depth 4 and high-value multiply remains
  `0%`.
- The remaining bottleneck is an explicit high-range product/value state and
  its readout, not a generic router. The next experiment will add a compact
  high-range semantic value codec/transition and keep V0.303 as the matched
  learned control.

## Artifacts

- Config:
  `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_operator_conditioned_product.yaml`
- Seed17 5k run:
  `results/runs/v0_303_operation_conditioned_product_seed17_5000.json`
- Seed18 5k run:
  `results/runs/v0_303_operation_conditioned_product_seed18_5000.json`
- Seed17/18 5k fixed operation results:
  `results/operationwise_fixed_checkpoint_eval_v0_303_seed17_5000.json`
  and `results/operationwise_fixed_checkpoint_eval_v0_303_seed18_5000.json`
- Seed17/18 high-value results:
  `results/operationwise_high_value_checkpoint_eval_v0_303_seed17_5000.json`
  and `results/operationwise_high_value_checkpoint_eval_v0_303_seed18_5000.json`
