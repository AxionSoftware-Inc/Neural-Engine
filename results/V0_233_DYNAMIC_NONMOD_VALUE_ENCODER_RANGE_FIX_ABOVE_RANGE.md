# V0.233 — corrected above-range representation test

**Date:** 2026-09-11  
**Status:** `PRELIMINARY RESULT; FOUR-SEED VALIDATION IN PROGRESS`

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
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- same factorized bank, active-8 route, and compact evaluator.

The treatment must beat the matched control on hard above-range accuracy and
depth-4 accuracy before it can be considered a generalization improvement.
The earlier V0.232 metrics are not part of this comparison.

## Preliminary results: seeds 17 and 18

| Arm | Seed | Train accuracy | Above-range accuracy | Depth 3 | Depth 4 | CE |
|---|---:|---:|---:|---:|---:|---:|
| No interaction | 17 | 98.24% | 56.84% | 61.33% | 52.34% | 11.2941 |
| No interaction | 18 | 99.61% | 59.57% | 66.41% | 52.73% | 10.4912 |
| **No interaction mean** |  | **98.93%** | **58.20%** | **63.87%** | **52.54%** | **10.8927** |
| Interaction rank 16 | 17 | 98.24% | 57.81% | 64.06% | 51.56% | 11.8264 |
| Interaction rank 16 | 18 | 99.41% | 60.74% | 66.41% | 55.08% | 10.6533 |
| **Interaction rank 16 mean** |  | **98.83%** | **59.28%** | **65.23%** | **53.32%** | **11.2399** |

Preliminary treatment-minus-control deltas are `+1.07 pp` overall,
`+1.37 pp` depth-3, `+0.78 pp` depth-4, and `+0.3472` CE (worse). Overall
accuracy is positive in both seeds, but depth-4 is mixed and the gain is
small; this is not a default-adoption result. Seeds19/20 are being added
before making the final decision.

## Raw runs

Four-seed completion pending. Treatment config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_value128.yaml`.

Control config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_value128.yaml`.
