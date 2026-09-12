# V0.290 — Learned numeric cross-digit multiplication bridge

**Date:** 2026-09-12  
**Branch:** `exp/track-native-engine`  
**Status:** `RETAINED AS STRONG OPT-IN CANDIDATE; REJECTED FOR DEFAULT ADOPTION`

## Question

V0.289 added cross-digit products in learned slot-embedding space and gave a
small but consistent gain. The remaining concern is that arbitrary learned
embedding coordinates do not necessarily preserve digit magnitude. V0.290
keeps the same cross-digit architecture but reads each typed slot through its
learned digit head, forms a soft expected digit value, and convolves those
values across positions. The resulting normalized partial-product scalar is
projected back into the learned transition space.

This is a learned numeric bridge, not an exact multiplication table: the slot
classifiers and projection are trained, and no arithmetic decoder or teacher is
used at inference.

## Protocol

- Same 300M virtual-bank factorized model and active-8 route as V0.288/V0.289;
- eight base-16 output digits, typed 8×16D carry state, contract loss `0.5`;
- numeric soft partial-product convolution enabled only for multiply;
- values `0..95`, train depths `1..2`, held-out depths `3..4`;
- 5,000 fresh steps, batch `128`, 1,024 examples per held-out depth;
- CUDA, seeds `17/18`, no value curriculum and no teacher forcing.

## Results

| Variant | Seed | Held-out acc | CE | Depth 3 | Depth 4 | Train sec |
|---|---:|---:|---:|---:|---:|---:|
| Typed carry + numeric convolution | 17 | 21.0449% | 12.1085 | 27.5391% | 14.5508% | 1076.9 |
| Typed carry + numeric convolution | 18 | 19.9707% | 11.4341 | 25.7813% | 14.1602% | 1064.9 |
| **Two-seed mean** | — | **20.5078%** | **11.7713** | **26.6602%** | **14.3555%** | **1070.9** |

Reference means from the matched V0.288/V0.289 screens:

| Variant | Held-out acc | CE | Depth 3 | Depth 4 |
|---|---:|---:|---:|---:|
| Base-16 control, plain | 20.7764% | 12.6073 | 26.4160% | 15.1367% |
| Typed carry, plain | 18.0176% | 12.0666 | 23.7305% | 12.3047% |
| Typed carry + embedding convolution | 19.0430% | 11.9985 | 24.7559% | 13.3301% |
| Typed carry + numeric convolution | 20.5078% | 11.7713 | 26.6602% | 14.3555% |

Numeric convolution improves typed-carry plain by `+2.4902 pp` overall,
`+2.9297 pp` at depth 3, and `+2.0508 pp` at depth 4; CE improves by
`−0.2953`. It improves over V0.289 embedding convolution by `+1.4648 pp` and
`−0.2272` CE. It still trails the simplest plain control by `−0.2686 pp`
overall and `−0.7813 pp` at depth 4, so it does not establish a better
architecture. Both seeds improve over the typed-carry baseline.

The numeric branch has `2,051,136` estimated active parameters, only `3,840`
above typed carry, but takes roughly `12.5%` more training time than typed
carry plain. Inference remains sparse: only the selected 8 circuits run; the
bridge is shared transition computation.

## Interpretation and decision

This is the strongest evidence so far that the capacity problem includes a
numeric multiply dataflow issue rather than only router quality. Turning the
learned typed state into soft numeric digits makes the cross-digit convolution
more useful and nearly closes the gap to the plain control. However, the simple
control remains the hard-accuracy leader, the depth-4 gap remains, and the
result is below the project `+2 pp` adoption gate against the control.

**Decision:** retain V0.290 as the leading opt-in Native candidate, but do not
change the default and do not scale to 700M/1B yet. P-003/P-004 remain active.
The next test is the same numeric bridge with the progressive curriculum, then
an operation-wise multiply evaluation. If it does not beat the plain control
reliably, the current typed-state family should be closed rather than widened.

## Artifacts

- `neural_engine/dynamic_register.py`
- `train_dynamic_composition.py`
- `tests/test_dynamic_register.py`
- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution.yaml`
- raw reports under `results/runs/v0_290_base16_numeric_multiply_convolution_*_5000.json`

