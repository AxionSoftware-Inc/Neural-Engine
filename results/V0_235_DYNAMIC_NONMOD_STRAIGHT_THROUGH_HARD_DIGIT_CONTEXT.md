# V0.235 — straight-through hard digit context screen

**Date:** 2026-09-11  
**Status:** `RUN CONFIGURED; RESULTS PENDING`

## Question

The current cross-digit interaction sends a soft probability mixture from one
digit head to the next. This can improve hard accuracy while worsening CE.
This opt-in variant sends a straight-through hard argmax in the forward pass,
while retaining the soft probability gradient in backpropagation. The test
asks whether sharper digit composition improves hard selection without the CE
regression seen in V0.233.

## Protocol

- `output_digit_context_mode: straight_through_hard`;
- rank-16 cross-digit interaction treatment;
- matched no-interaction control;
- learned value encoder with `value_encoder_modulus=128`;
- polynomial2/Fourier algebraic state;
- four base-512 output digits, shared output rank `128`;
- train operands `0..63`, held-out operands `64..95`;
- train depths `1..2`, eval depths `3..4`;
- target offset `134,217,728`, `num_classes=8,589,934,592`;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- same factorized bank, active-8 route, and compact evaluator.

The treatment must beat the matched control on hard above-range and depth-4
accuracy, and should not regress against the corrected soft-context rank16
baseline in V0.233.

## Raw runs

Pending completion. Treatment config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_hardcontext_value128.yaml`.

Control config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_hardcontext_value128.yaml`.
