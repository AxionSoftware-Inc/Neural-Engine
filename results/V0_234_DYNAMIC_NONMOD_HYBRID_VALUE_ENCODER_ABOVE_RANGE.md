# V0.234 — hybrid Fourier value encoder above-range screen

**Date:** 2026-09-11  
**Status:** `RUN CONFIGURED; RESULTS PENDING`

## Question

V0.233 fixed the non-modular input range and established a small but consistent
rank16 hard-accuracy gain on `0..63 → 64..95`. CE still regressed, so this
screen tests whether a fixed Fourier basis can improve the extrapolative input
signal without discarding the adaptable learned projection. The interaction
and no-interaction arms are both rerun under the same hybrid encoder.

## Protocol

- `value_encoder_mode: hybrid_fourier`;
- `value_encoder_modulus: 128`;
- treatment: rank-16 cross-digit interaction;
- control: matched no-interaction output head;
- polynomial2/Fourier algebraic state;
- four base-512 output digits, shared output rank `128`;
- train operands `0..63`, held-out operands `64..95`;
- train depths `1..2`, eval depths `3..4`;
- target offset `134,217,728`, `num_classes=8,589,934,592`;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- same factorized bank, active-8 route, and compact evaluator.

The hybrid treatment must beat its matched hybrid control and should also
avoid regressing against the corrected learned-encoder baseline in V0.233.
This is an opt-in representation screen, not a default change or capacity
scaling experiment.

## Raw runs

Pending completion. Treatment config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_hybrid_value128.yaml`.

Control config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_hybrid_value128.yaml`.
