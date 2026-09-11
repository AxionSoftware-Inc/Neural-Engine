# V0.235 — straight-through hard digit context screen

**Date:** 2026-09-11  
**Status:** `PRELIMINARY SMALL GAIN; DEFAULT UNCHANGED`

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

The four fresh runs completed successfully. The matched no-interaction
controls reproduce the ordinary soft-context control because no cross-digit
context is consumed when the interaction rank is zero.

| seed | arm | held-out | depth-3 | depth-4 | CE |
|---:|---|---:|---:|---:|---:|
| 17 | no interaction control | 56.8359% | 61.3281% | 52.3438% | 11.294106 |
| 17 | hard-context rank16 | 58.5938% | 65.6250% | 51.5625% | 11.390585 |
| 18 | no interaction control | 59.5703% | 66.4063% | 52.7344% | 10.491230 |
| 18 | hard-context rank16 | 60.7422% | 67.1875% | 54.2969% | 10.283558 |

Two-seed means are `58.2031% → 59.6680%` held-out
(`+1.4648 pp`), `63.8672% → 66.4063%` depth-3 (`+2.5391 pp`), and
`52.5391% → 52.9297%` depth-4 (`+0.3906 pp`). Mean CE changes from
`10.892668` to `10.837071` (`−0.055597`, better). Per-seed depth-4 is not
uniform: seed17 falls by `0.7813 pp`, while seed18 rises by `1.5625 pp`.

The result is a small positive signal for sharper digit composition, not a
capacity result or a reliable solution to the scaling problem. It does not
meet the project's adoption bar for a default architecture change. Keep the
implementation and configs as an opt-in diagnostic; leave the default model,
router, and 300M capacity unchanged. Do not move to 700M/1B based on this
two-seed screen.

Treatment config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_hardcontext_value128.yaml`.

Control config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_hardcontext_value128.yaml`.

Raw result files:

- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_hardcontext_value128_seed17_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_hardcontext_value128_seed18_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_hardcontext_value128_seed17_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_hardcontext_value128_seed18_5000.json`
