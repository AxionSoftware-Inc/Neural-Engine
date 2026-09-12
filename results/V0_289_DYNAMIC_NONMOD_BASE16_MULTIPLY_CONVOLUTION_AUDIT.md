# V0.289 — Cross-digit multiplication convolution

**Date:** 2026-09-12  
**Branch:** `exp/track-native-engine`  
**Status:** `RETAINED AS DIAGNOSTIC; REJECTED FOR QUALITY ADOPTION`

## Question

The V0.286/V0.288 typed carry transition processes matching digit slots
(`state_i`, `operand_i`) and passes a carry packet. That is a plausible local
inductive bias for addition, but multiplication needs cross-digit products:
the output position `k` depends on pairs `(i, k-i)`. V0.289 adds a shared
least-significant-first cross-digit convolution to the multiply transition only.

For every multiply step, the new feature forms

```text
product[k] = sum(state[i] * operand[k-i])
```

in learned 16D slot space. A separate small shared transition consumes this
product packet together with the ordinary local slot/carry inputs. Add and
subtract continue using the original carry transition. No exact arithmetic
table, integer decoder, teacher forcing, attention, or dense fallback is added.

## Protocol

- Same 300M virtual-bank factorized model and active-8 route as V0.288;
- eight base-16 output digits, typed 8×16D carry state, contract loss `0.5`;
- values `0..95`, train depths `1..2`, held-out depths `3..4`;
- 5,000 fresh steps, batch `128`, 1,024 examples per held-out depth;
- CUDA, seeds `17/18`, no value curriculum and no teacher forcing;
- comparison is against V0.288's matched typed-carry plain arm and the plain
  base-16 control.

## Results

| Variant | Seed | Held-out acc | CE | Depth 3 | Depth 4 | Train sec |
|---|---:|---:|---:|---:|---:|---:|
| Typed carry, plain (V0.288) | 17 | 18.6523% | 12.1728 | 24.3164% | 12.9883% | 952.6 |
| Typed carry + multiply convolution | 17 | 19.6289% | 11.7624 | 25.5859% | 13.6719% | 1024.6 |
| Typed carry, plain (V0.288) | 18 | 17.3828% | 11.9605 | 23.1445% | 11.6211% | 951.1 |
| Typed carry + multiply convolution | 18 | 18.4570% | 12.2346 | 23.9258% | 12.9883% | 1071.0 |
| Base-16 control, plain (V0.288) | 17 | 22.4609% | 12.2262 | 28.1250% | 16.7969% | 774.4 |
| Base-16 control, plain (V0.288) | 18 | 19.0918% | 12.9883 | 24.7070% | 13.4766% | 728.4 |

Two-seed means:

| Variant | Held-out acc | CE | Depth 3 | Depth 4 | Train sec |
|---|---:|---:|---:|---:|---:|
| Typed carry, plain | 18.0176% | 12.0666 | 23.7305% | 12.3047% | 951.9 |
| Typed carry + multiply convolution | 19.0430% | 11.9985 | 24.7559% | 13.3301% | 1047.8 |
| Base-16 control, plain | 20.7764% | 12.6073 | 26.4160% | 15.1367% | 751.4 |

Relative to typed-carry plain, the convolution gives `+1.0254 pp` accuracy,
`−0.0681` CE, `+1.0254 pp` at depth 3, and `+1.0254 pp` at depth 4. Both
seeds improve overall accuracy. However, it remains `−1.7334 pp` below the
plain base-16 control and costs about `10.1%` more training time than typed
carry. It adds only `3,808` parameters to the active estimate (`2,051,104`
versus `2,047,296`), and inference still selects only the configured 8
circuits.

## Interpretation and decision

This is stronger evidence for the architectural diagnosis than the previous
typed-carry screens: adding the missing cross-digit interaction improves both
seeds and both held-out depths. It is still not a solution because the best
typed variant cannot beat the simpler base-16 control, and the gain is below
the project `+2 pp` adoption gate. The convolution is a learned feature-space
interaction, not an exact multiplication algorithm; carry normalization and
partial-product accumulation remain unresolved.

**Decision:** keep V0.289 opt-in and reject it as the default quality path or a
reason to scale to 700M/1B. P-003/P-004 remain active. The next Native test
should isolate a compact partial-product/carry accumulator or operation-wise
multiply transfer, rather than increasing router capacity or repeating a
generic state-width sweep.

## Artifacts

- `neural_engine/dynamic_register.py`
- `train_dynamic_composition.py`
- `tests/test_dynamic_register.py`
- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_convolution.yaml`
- raw reports under `results/runs/v0_289_base16_multiply_convolution_*_5000.json`

