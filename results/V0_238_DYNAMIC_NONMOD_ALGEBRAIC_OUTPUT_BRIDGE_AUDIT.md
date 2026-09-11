# V0.238 — direct algebraic packet output bridge

**Date:** 2026-09-11  
**Status:** `REJECTED; MULTIPLY UNCHANGED`

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

The two fresh runs completed with unchanged total and active parameter
counts. On the matched random held-out evaluation, the baseline-to-bridge
means are `58.2031% → 58.7891%` overall, `63.8672% → 63.6719%`
depth-3, and `52.5391% → 53.9063%` depth-4; mean CE worsens from
`10.892668` to `11.625465` (`+0.732797`). The apparent hard-accuracy change
is small and seed-matched task-wise evaluation is more revealing:

| operation | control held-out | bridge held-out | delta |
|---|---:|---:|---:|
| add | 100.00% | 100.00% | 0.00 pp |
| subtract | 55.7617% | 60.2539% | +4.4922 pp |
| multiply | 0.00% | 0.00% | 0.00 pp |

The bridge adds no parameters and does not create any multiply correctness;
the remaining signal is again subtract-specific. **REJECTED FOR ADOPTION.**
The exact packet is present, but the current above-range protocol asks the
model to decode products outside the product range seen during training. The
next control therefore widens the training operand range before changing the
state architecture.
