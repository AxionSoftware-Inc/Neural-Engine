# V0.234 — hybrid Fourier value encoder above-range screen

**Date:** 2026-09-11  
**Status:** `REJECTED; HYBRID DOES NOT IMPROVE INTERACTION ARM`

## Question

V0.233 fixed the non-modular input range and established a small but consistent
rank16 hard-accuracy gain on `0..63 → 64..95`. CE still regressed, so this
screen tests whether a fixed Fourier basis can improve the extrapolative input
signal without discarding the adaptable learned projection. The interaction
and no-interaction arms are both rerun under the same hybrid encoder.

## Protocol

- `value_encoder_mode: hybrid_fourier`;
- `value_encoder_modulus: 128`;
- treatment: rank-16 cross-digit interaction;
- control: matched no-interaction output head;
- polynomial2/Fourier algebraic state;
- four base-512 output digits, shared output rank `128`;
- train operands `0..63`, held-out operands `64..95`;
- train depths `1..2`, eval depths `3..4`;
- target offset `134,217,728`, `num_classes=8,589,934,592`;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- same factorized bank, active-8 route, and compact evaluator.

The hybrid treatment must beat its matched hybrid control and should also
avoid regressing against the corrected learned-encoder baseline in V0.233.
This is an opt-in representation screen, not a default change or capacity
scaling experiment.

## Results: seeds 17 and 18

| Arm | Seed | Train accuracy | Above-range accuracy | Depth 3 | Depth 4 | CE |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid no interaction | 17 | 98.44% | 56.84% | 62.89% | 50.78% | 11.5744 |
| Hybrid no interaction | 18 | 99.02% | 60.74% | 67.97% | 53.52% | 9.9660 |
| **Hybrid control mean** |  | **98.73%** | **58.79%** | **65.43%** | **52.15%** | **10.7702** |
| Hybrid interaction rank 16 | 17 | 98.24% | 58.01% | 64.45% | 51.56% | 12.4079 |
| Hybrid interaction rank 16 | 18 | 99.22% | 58.79% | 64.45% | 53.13% | 11.4894 |
| **Hybrid interaction mean** |  | **98.73%** | **58.40%** | **64.45%** | **52.34%** | **11.9486** |

Treatment-minus-control deltas are `−0.39 pp` overall, `−0.98 pp`
depth-3, `+0.20 pp` depth-4, and `+1.1784` CE (worse). Against the corrected
learned-value128 rank16 treatment in V0.233, hybrid also drops overall and
depth-4 hard accuracy. The fixed basis therefore does not provide a useful
interaction enhancement under this budget.

## Decision

Reject `hybrid_fourier` for adoption in the current above-range path. Keep the
corrected learned-value128 rank16 model as the leading hard-quality opt-in;
the default remains unchanged. This result does not close P-003, but it rules
out another low-cost input representation tweak before any capacity scaling.

## Raw runs

Treatment config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_hybrid_value128.yaml`.

Treatment runs:

- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_hybrid_value128_seed17_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_interaction16_hybrid_value128_seed18_5000.json`

Control config:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_hybrid_value128.yaml`.

Control runs:

- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_hybrid_value128_seed17_5000.json`
- `results/runs/nonmod_train0_63_eval64_95_four_digit_base512_rank128_nointeraction_hybrid_value128_seed18_5000.json`
