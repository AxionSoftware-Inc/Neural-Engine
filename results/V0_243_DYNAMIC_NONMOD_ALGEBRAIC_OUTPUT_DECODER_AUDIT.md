# V0.243 — direct algebraic packet-to-digit decoder

**Date:** 2026-09-11  
**Status:** `REJECTED FOR MULTIPLY QUALITY ADOPTION`

## Question

V0.242 showed that replacing the learned query state with the algebraic
packet does not repair multiply. This experiment isolates the final readout:
the recurrent learned state is bypassed and a separate small learned decoder
maps the existing polynomial2/Fourier packet directly to the four digit
heads. The sparse recurrent body and router remain unchanged.

## Protocol

- V0.240 hard-context rank-16 interaction body;
- `algebraic_output_decoder=true`, using the existing packet features;
- operands `0..95`, train depths `1..2`, held-out depths `3..4`;
- target offset `134,217,728`, four-digit base-512 compact evaluator;
- 5000 steps, batch `128`, seeds `17` and `18`;
- task-wise held-out comparison against V0.240 hard context.

## Results

| Metric | V0.240 hard context | V0.243 direct decoder | Delta |
|---|---:|---:|---:|
| All-operations accuracy | 82.7148% | 85.1563% | +2.4414 pp |
| Depth-3 accuracy | 86.5234% | 89.2578% | +2.7344 pp |
| Depth-4 accuracy | 78.9063% | 81.0547% | +2.1484 pp |
| All-operations CE | 2.682122 | 2.649375 | -0.032747 |

The operation-wise breakdown shows why this is not a multiply breakthrough:

| Operation | V0.240 hard context | V0.243 direct decoder | Delta |
|---|---:|---:|---:|
| Add | 100.0000% | 100.0000% | +0.0000 pp |
| Subtract | 99.7070% | 100.0000% | +0.2930 pp |
| Multiply | 15.4297% | 13.5742% | -1.8555 pp |

Multiply depth-3 falls `23.0469% → 20.3125%` and depth-4 falls
`7.8125% → 6.8359%`. The aggregate gain is therefore entirely explained by
the already easy add/subtract tasks. The direct decoder's extra parameters
are `17,368`; total parameters are `7,518,111`.

## Decision

The direct floating-point packet decoder is **REJECTED FOR MULTIPLY QUALITY
ADOPTION**. It does not prove that the recurrent body is the only bottleneck;
the current normalized float packet can lose low-order information when
products exceed float32's exact-integer range, and its Fourier basis may alias
large products. The next diagnostic preserves the exact integer value in a
separate non-gradient register and learns a base-512 digit decoder from that
lossless packet. This tests numeric precision/coding without increasing the
sparse circuit bank or scaling capacity.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_outputdecoder.yaml`
- `results/runs/v0_243_wide_range_outputdecoder_seed17_5000.json`
- `results/runs/v0_243_wide_range_outputdecoder_seed18_5000.json`
- `results/runs/v0_243_wide_range_outputdecoder_taskwise_seed17.json`
- `results/runs/v0_243_wide_range_outputdecoder_taskwise_seed18.json`
