# V0.232 — rank-16 interaction above-range extrapolation

**Date:** 2026-09-11  
**Status:** `INVALIDATED; INPUT ENCODING COLLAPSED ABOVE-RANGE VALUES`

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

At the time of this run, the non-modular input encoder still used its legacy
64-value range. Consequently, raw operands `64..95` were not represented as
distinct numeric inputs. This is an input/benchmark validity failure, not an
architecture result.

## Protocol

- rank-16 cross-digit interaction;
- learned value encoder and polynomial2/Fourier algebraic state;
- four base-512 output digits, shared output rank `128`;
- train operands `0..63`, held-out operands `64..95`;
- train depths `1..2`, eval depths `3..4`;
- target offset `134,217,728`;
- `num_classes=8,589,934,592` for the above-range target envelope;
- legacy effective input encoder range `0..63` (the defect);
- 5000 steps, batch `128`, seeds `17` and `18`;
- same factorized bank, active-8 route, and compact evaluator.

The comparison was made with a matched no-interaction control, but no quality
conclusion can be drawn because both arms received collapsed above-range
inputs. V0.233 reruns the screen with `value_encoder_modulus=128`.

## Preflight note

The first preflight run used the earlier `num_classes=1,073,741,824` envelope
and correctly stopped during final evaluation because an above-range target
reached about `3.75B`. It produced no valid metrics and is excluded from the
comparison. After increasing the class envelope, the run completed, but the
input encoder defect was then identified; those completed metrics are retained
below only for auditability and are excluded from model comparison.

## Invalid run diagnostics (excluded)

| Arm | Seed | Train accuracy | Above-range accuracy | Depth 3 | Depth 4 | CE |
|---|---:|---:|---:|---:|---:|---:|
| No interaction | 17 | 98.05% | 50.00% | 55.08% | 44.92% | 11.9260 |
| No interaction | 18 | 99.41% | 50.78% | 59.77% | 41.80% | 9.5744 |
| **No interaction mean** |  | **98.73%** | **50.39%** | **57.42%** | **43.36%** | **10.7502** |
| Interaction rank 16 | 17 | 97.85% | 51.95% | 56.25% | 47.66% | 11.7484 |
| Interaction rank 16 | 18 | 99.02% | 45.51% | 53.91% | 37.11% | 9.4201 |
| **Interaction rank 16 mean** |  | **98.44%** | **48.73%** | **55.08%** | **42.38%** | **10.5843** |

Matched treatment minus control deltas:

- above-range accuracy: `−1.66 pp`;
- depth-3 accuracy: `−2.34 pp`;
- depth-4 accuracy: `−0.98 pp`;
- CE: `−0.1660` (numerically improved, but hard accuracy declined).

The per-seed hard-accuracy deltas are mixed: seed17 is positive (`+1.95 pp`
overall, `+2.73 pp` depth-4), while seed18 is negative (`−5.27 pp`,
`−4.69 pp`). These numbers are not evidence for or against interaction,
because the above-range inputs were not distinct at the model boundary.

This does not invalidate the earlier `0..31 → 32..63` rank16 result. It only
shows that the attempted `0..63 → 64..95` screen must be rerun after fixing
the input representation.

## Decision

V0.232 is **invalidated as a model comparison**. The default is unchanged and
no interaction decision is made from these runs. The concrete defect is fixed
by the explicit `value_encoder_modulus` parameter; V0.233 is the valid
follow-up. Do not scale to 700M/1B from an invalid screen.

## Raw runs

Treatment:

- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_above_range_seed17_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_above_range_seed18_5000.json`

Control:

- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_above_range_seed17_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_above_range_seed18_5000.json`

Invalid-run config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_above_range.yaml`.

Matched control config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_above_range.yaml`.

Corrected implementation and rerun:
`neural_engine/encoding.py`, `neural_engine/dynamic_register.py`, and
`results/V0_233_DYNAMIC_NONMOD_VALUE_ENCODER_RANGE_FIX_ABOVE_RANGE.md`.
