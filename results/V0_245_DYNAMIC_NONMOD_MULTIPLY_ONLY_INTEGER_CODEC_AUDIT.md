# V0.245 — multiply-only exact integer output codec

**Date:** 2026-09-11  
**Status:** `REJECTED AS END-TO-END TRAINING CONFIGURATION`

## Question

V0.244 proved that a lossless integer packet helps multiply but damages
subtract when used for every operation. V0.245 selects the integer output
decoder only when the terminal operation is multiply and uses the learned
state readout for add/subtract.

## Protocol

- same V0.240 clean wide-support/depth-holdout body;
- `algebraic_integer_output_decoder=true`;
- `algebraic_integer_output_decoder_mode=multiply_only`;
- all model parameters trained from a fresh initialization;
- operands `0..95`, train depths `1..2`, held-out depths `3..4`;
- target offset `134,217,728`, 5000 steps, batch `128`, seeds `17` and `18`.

## Results

| Metric | V0.240 hard context | V0.245 multiply-only | Delta |
|---|---:|---:|---:|
| All-operations accuracy | 82.7148% | 66.3086% | -16.4063 pp |
| Depth-3 accuracy | 86.5234% | 73.2422% | -13.2813 pp |
| Depth-4 accuracy | 78.9063% | 59.3750% | -19.5313 pp |
| All-operations CE | 2.682122 | 7.873952 | +5.191830 |

Task-wise held-out means:

| Operation | V0.240 hard context | V0.245 multiply-only | Delta |
|---|---:|---:|---:|
| Add | 100.0000% | 99.7070% | -0.2930 pp |
| Subtract | 99.7070% | 79.6875% | -20.0195 pp |
| Multiply | 15.4297% | 22.5586% | +7.1289 pp |

Multiply depth-3 rises `23.0469% → 35.9375%` (`+12.8906 pp`) and depth-4
rises `7.8125% → 9.1797%` (`+1.3672 pp`). The multiply signal survives,
but the shared body/output training becomes unstable and the subtract path
collapses, especially on seed18.

## Decision

The end-to-end multiply-only configuration is **REJECTED**. The result does
not invalidate the exact integer signal; it shows that operation-conditioned
decoding must be added without re-training or disturbing the already learned
body. The next experiment loads a V0.240 hard-context checkpoint, freezes all
existing parameters, and trains only the new integer decoder overlay. This
should preserve add/subtract by construction while testing whether multiply
can be improved safely.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_integerdecoder.yaml`
- `results/runs/v0_245_wide_range_integerdecoder_multiplyonly_seed17_5000.json`
- `results/runs/v0_245_wide_range_integerdecoder_multiplyonly_seed18_5000.json`
- `results/runs/v0_245_wide_range_integerdecoder_multiplyonly_taskwise_seed17.json`
- `results/runs/v0_245_wide_range_integerdecoder_multiplyonly_taskwise_seed18.json`
