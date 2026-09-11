# V0.238 — direct algebraic packet output bridge

**Date:** 2026-09-11  
**Status:** `RUN CONFIGURED; RESULTS PENDING`

## Question

V0.236 localized the above-range ceiling to multiply, and V0.237 showed that
conditioning the learned output readout on the operation ID does not restore
multiply accuracy. The current model already carries an exact polynomial2 /
Fourier value packet, but it is first mixed into the recurrent state and then
normalized before the digit codec. This opt-in screen reuses the existing
learned packet projection as a direct additive bridge after the output
LayerNorm. No new parameters, circuit bank rows, router paths, or arithmetic
operations are added.

## Protocol

- no-interaction four-digit base-512 rank-128 control;
- `algebraic_output_bridge_scale=1.0` treatment;
- learned value encoder with `value_encoder_modulus=128`;
- polynomial2/Fourier algebraic state;
- train operands `0..63`, held-out operands `64..95`;
- train depths `1..2`, eval depths `3..4`;
- target offset `134,217,728`, compact factorized evaluator;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- primary readout: held-out multiply accuracy and CE; add/subtract are
  regression controls.

The bridge is retained only if it produces repeatable multiply accuracy or a
clear hard-quality gain without destabilizing the other operations. Since it
adds no parameters, a negative result is strong evidence that the bottleneck
is not merely parameter capacity in the output head.

## Results

Pending completion.
