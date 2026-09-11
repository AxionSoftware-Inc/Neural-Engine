# V0.239 — wide training-range multiply control

**Date:** 2026-09-11  
**Status:** `RUN CONFIGURED; RESULTS PENDING`

## Question

The corrected above-range protocol trains on operands `0..63` and evaluates
on `64..95`. V0.236–V0.238 show that multiply remains at `0%` while add and
subtract can generalize. Before changing the recurrent state again, this
control asks whether multiply is simply unsupported because products from
`64..95` (up to `9025`) never occur during training.

## Protocol

- base four-digit rank-128 no-interaction model; no new architecture;
- learned value encoder with `value_encoder_modulus=128`;
- polynomial2/Fourier algebraic state;
- training operands `0..95`, evaluation operands `0..95`;
- train depths `1..2`, held-out depths `3..4`;
- target offset `134,217,728`, compact factorized evaluator;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- task-wise follow-up for add/subtract/multiply.

Interpretation:

- if multiply becomes non-zero, the main failure is product-range support/OOD;
- if multiply remains near zero, the recurrent composition or value-to-output
  dataflow is the stronger suspect;
- this result does not authorize 700M/1B scaling.

## Results

Pending completion.
