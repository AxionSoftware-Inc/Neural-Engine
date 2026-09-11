# V0.244 — exact integer packet output codec

**Date:** 2026-09-11  
**Status:** `STRONG MULTIPLY DIAGNOSTIC; NOT DEFAULT`

## Question

V0.243's floating-point packet decoder did not improve multiply. The
normalized packet can lose low-order integer bits for products beyond the
float32 exact-integer range. V0.244 therefore keeps a separate exact int64
register and feeds learned base-512 digit embeddings to the output decoder.
The recurrent sparse body and router are unchanged.

## Protocol

- V0.240 hard-context rank-16 interaction body;
- exact integer state updated by the same add/subtract/multiply program;
- four learned base-512 digit embeddings plus sign feature;
- exact-integer decoder used for all operations in this first screen;
- operands `0..95`, train depths `1..2`, held-out depths `3..4`;
- target offset `134,217,728`, 5000 steps, batch `128`, seeds `17` and `18`.

## Results

| Metric | V0.240 hard context | V0.244 integer codec | Delta |
|---|---:|---:|---:|
| All-operations accuracy | 82.7148% | 86.0352% | +3.3203 pp |
| Depth-3 accuracy | 86.5234% | 89.2578% | +2.7344 pp |
| Depth-4 accuracy | 78.9063% | 82.8125% | +3.9063 pp |
| All-operations CE | 2.682122 | 1.614753 | -1.067369 |

The operation-wise breakdown is the key result:

| Operation | V0.240 hard context | V0.244 integer codec | Delta |
|---|---:|---:|---:|
| Add | 100.0000% | 99.8047% | -0.1953 pp |
| Subtract | 99.7070% | 80.1758% | -19.5313 pp |
| Multiply | 15.4297% | 23.0469% | +7.6172 pp |

Multiply depth-3 rises `23.2422% → 36.7188%` (`+13.4766 pp`) and depth-4
rises `7.6172% → 9.3750%` (`+1.7578 pp`). The all-operation aggregate
improvement is therefore real but partially hides a large subtract regression.
The sign/magnitude digit representation is adequate for the multiply
diagnostic but is not a universal codec for the mixed operation distribution.

The integer decoder adds `58,242` parameters only through small digit
embeddings and its projection; the exact run reports `7,558,985` total
parameters. It
does not add circuit-bank capacity or active routed circuits.

## Decision

This is a **strong multiply-specific diagnostic**, not a default architecture.
It establishes that low-order numeric representation is part of the P-003
ceiling, while also showing that one codec should not be forced on every
operation. The next controlled test uses the exact integer decoder only for
multiply and retains the learned state readout for add/subtract. No 700M/1B
scaling follows yet.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_integerdecoder.yaml`
- `results/runs/v0_244_wide_range_integerdecoder_seed17_5000.json`
- `results/runs/v0_244_wide_range_integerdecoder_seed18_5000.json`
- `results/runs/v0_244_wide_range_integerdecoder_taskwise_seed17.json`
- `results/runs/v0_244_wide_range_integerdecoder_taskwise_seed18.json`
