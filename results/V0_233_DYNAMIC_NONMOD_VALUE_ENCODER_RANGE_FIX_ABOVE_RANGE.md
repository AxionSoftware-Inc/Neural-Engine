# V0.233 — corrected above-range representation test

**Date:** 2026-09-11  
**Status:** `VALIDATED SMALL HARD-QUALITY GAIN; DEFAULT UNCHANGED`

## Motivation

V0.232 was invalidated before interpreting its metrics. In non-modular mode,
`DynamicRegisterNeuralEngine` still passed the legacy `VALUE_MODULUS=64` to
`encode_tokens`. Therefore operand tokens for raw values `64..95` did not
receive distinct numeric encodings. The corrected test adds an explicit
`value_encoder_modulus=128`, which makes raw `0..95` values representable while
leaving the circuit bank, router, state, and output interaction unchanged.

## Protocol

- treatment: rank-16 cross-digit interaction;
- control: matched no-interaction output head;
- learned value encoder with `value_encoder_modulus=128`;
- polynomial2/Fourier algebraic state;
- four base-512 output digits, shared output rank `128`;
- train operands `0..63`, held-out operands `64..95`;
- train depths `1..2`, eval depths `3..4`;
- target offset `134,217,728`;
- `num_classes=8,589,934,592` (`2^33`) for the full non-modular target
  envelope, still represented by compact factorized digit heads;
- fresh 5000-step runs, batch `128`, seeds `17`, `18`, `19`, and `20`;
- same factorized bank, active-8 route, and compact evaluator.

The treatment must beat the matched control on hard above-range accuracy and
depth-4 accuracy before it can be considered a generalization improvement.
The earlier V0.232 metrics are not part of this comparison.

## Results

| Arm | Seed | Train accuracy | Above-range accuracy | Depth 3 | Depth 4 | CE |
|---|---:|---:|---:|---:|---:|---:|
| No interaction | 17 | 98.24% | 56.84% | 61.33% | 52.34% | 11.2941 |
| No interaction | 18 | 99.61% | 59.57% | 66.41% | 52.73% | 10.4912 |
| No interaction | 19 | 98.44% | 56.25% | 69.14% | 43.36% | 11.7247 |
| No interaction | 20 | 98.63% | 55.86% | 62.50% | 49.22% | 11.3267 |
| **No interaction mean** |  | **98.73%** | **57.13%** | **64.84%** | **49.41%** | **11.2092** |
| Interaction rank 16 | 17 | 98.24% | 57.81% | 64.06% | 51.56% | 11.8264 |
| Interaction rank 16 | 18 | 99.41% | 60.74% | 66.41% | 55.08% | 10.6533 |
| Interaction rank 16 | 19 | 98.44% | 58.01% | 69.92% | 46.09% | 12.4176 |
| Interaction rank 16 | 20 | 99.22% | 55.86% | 62.89% | 48.83% | 11.4935 |
| **Interaction rank 16 mean** |  | **98.83%** | **58.11%** | **65.82%** | **50.39%** | **11.5977** |

Four-seed treatment-minus-control deltas are `+0.98 pp` overall,
`+0.98 pp` depth-3, `+0.98 pp` depth-4, and `+0.3885` CE (worse). Overall
hard accuracy is non-negative in all four seeds; depth-4 is mixed by seed,
with a positive four-seed mean. The hard-quality gain is real but small and
comes with a CE regression, so this is not a default-adoption result.

## Raw runs

Treatment config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_value128.yaml`.

Treatment runs:

- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_value128_seed17_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_value128_seed18_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_value128_seed19_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_value128_seed20_5000.json`

Control config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_value128.yaml`.

Control runs:

- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_value128_seed17_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_value128_seed18_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_value128_seed19_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_value128_seed20_5000.json`
