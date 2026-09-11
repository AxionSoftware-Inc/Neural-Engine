# V0.232 — rank-16 interaction above-range extrapolation

**Date:** 2026-09-11  
**Status:** `RUN CONFIGURED; RESULTS PENDING`

## Question

V0.231 validates rank16 on the unseen half-range `0..31 → 32..63`. This next
gate asks whether the same representation can extrapolate beyond the entire
training interval: train on operands `0..63`, then evaluate on `64..95`.

This is intentionally a value-range test, not a capacity test. The model,
router, circuit bank, output interaction rank, and active budget stay fixed.
The target offset is increased to `134,217,728` so unmodulated multiply and
subtract chains remain in the non-negative class range. Because four
operations can consume five operands, the upper value range also requires a
larger classifier class envelope: `num_classes=8,589,934,592` (`2^33`), which
is still a compact factorized head (`64 + 3*512` digit logits), not a dense
8.6-billion-logit layer.

## Planned protocol

- rank-16 cross-digit interaction;
- learned value encoder and polynomial2/Fourier algebraic state;
- four base-512 output digits, shared output rank `128`;
- train operands `0..63`, held-out operands `64..95`;
- train depths `1..2`, eval depths `3..4`;
- target offset `134,217,728`;
- `num_classes=8,589,934,592` for the above-range target envelope;
- 5000 steps, batch `128`, seeds `17` and `18`;
- same factorized bank, active-8 route, and compact evaluator.

The comparison must be made with a matched no-interaction control under the
same above-range split; no quality conclusion should be drawn from the
treatment alone.

## Raw runs

The first preflight run used the earlier `num_classes=1,073,741,824` envelope
and correctly stopped during final evaluation because an above-range target
reached about `3.75B`. It produced no valid metrics and is excluded from the
comparison. The config was corrected before the valid rerun.

Pending completion after the corrected rerun. Config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_above_range.yaml`.

Matched control config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_above_range.yaml`.
