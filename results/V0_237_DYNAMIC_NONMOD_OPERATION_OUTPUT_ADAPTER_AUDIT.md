# V0.237 — operation-conditioned output adapter screen

**Date:** 2026-09-11  
**Status:** `RUN CONFIGURED; RESULTS PENDING`

## Question

V0.236 showed that the current above-range multiply path is at `0%` in both
the hard-context treatment and its control, while hard context mainly helps
subtract. This screen adds a small operation-specific low-rank adapter to the
output readout. The adapter receives the last executed primitive operation;
shorter programs reuse that operation at padded terminal steps. The bank,
router, recurrent writer, and active-8 circuit budget remain unchanged.

## Protocol

- no-interaction four-digit base-512 rank-128 control;
- operation-output adapter rank `16`, scale `1.0`;
- learned value encoder with `value_encoder_modulus=128`;
- polynomial2/Fourier algebraic state;
- train operands `0..63`, held-out operands `64..95`;
- train depths `1..2`, eval depths `3..4`;
- target offset `134,217,728`, compact factorized evaluator;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- primary readout: held-out multiply accuracy and CE, with add/subtract as
  controls.

The adapter is retained only if it produces non-zero and repeatable multiply
accuracy without damaging add/subtract or materially inflating the active
path. This is an opt-in diagnostic, not a default change.

## Results

Pending completion.
