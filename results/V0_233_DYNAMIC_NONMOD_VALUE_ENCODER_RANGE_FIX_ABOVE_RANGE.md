# V0.233 — corrected above-range representation test

**Date:** 2026-09-11  
**Status:** `RUN CONFIGURED; RESULTS PENDING`

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

## Raw runs

Pending completion. Treatment config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_value128.yaml`.

Control config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_value128.yaml`.
