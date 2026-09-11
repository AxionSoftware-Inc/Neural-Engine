# V0.231 — four-seed rank-16 unseen-range validation

**Date:** 2026-09-11  
**Status:** `VALIDATED OOD OPT-IN; DEFAULT UNCHANGED`

## Question

V0.224 established a rank-32 interaction gain on the unseen-range split, and
V0.226/V0.230 showed rank16 is a strong lower-cost full-range candidate. This
audit completes the same unseen-range comparison with seeds19/20 so the OOD
signal is not based on only two seeds.

## Protocol

- treatment: `output_digit_interaction_rank=16`;
- control: `output_digit_interaction_rank=0`;
- learned value encoder and polynomial2/Fourier algebraic state;
- four base-512 output digits, shared output rank `128`;
- train operands `0..31`, unseen eval operands `32..63`;
- train depths `1..2`, eval depths `3..4`;
- safe target offset `33,554,432`;
- fresh 5000-step runs, batch `128`, seeds `17,18,19,20`;
- same factorized bank, active-8 route and compact evaluator.

## Results

| Arm | Seed | Train accuracy | Unseen accuracy | Depth 3 | Depth 4 | CE |
|---|---:|---:|---:|---:|---:|---:|
| No interaction | 17 | 99.80% | 55.66% | 59.38% | 51.95% | 12.6858 |
| No interaction | 18 | 100.00% | 56.84% | 62.11% | 51.56% | 12.1238 |
| No interaction | 19 | 99.80% | 55.27% | 67.58% | 42.97% | 13.2121 |
| No interaction | 20 | 99.80% | 53.91% | 63.67% | 44.14% | 12.8595 |
| **No interaction mean** |  | **99.85%** | **55.42%** | **63.18%** | **47.66%** | **12.7203** |
| Interaction rank 16 | 17 | 100.00% | 57.81% | 60.16% | 55.47% | 11.8653 |
| Interaction rank 16 | 18 | 100.00% | 59.96% | 64.84% | 55.08% | 11.7561 |
| Interaction rank 16 | 19 | 99.80% | 58.59% | 71.48% | 45.70% | 12.7104 |
| Interaction rank 16 | 20 | 99.80% | 56.45% | 66.80% | 46.09% | 13.0235 |
| **Interaction rank 16 mean** |  | **99.90%** | **58.20%** | **65.82%** | **50.59%** | **12.3388** |

Matched treatment minus control deltas:

- overall unseen accuracy: `+2.78 pp`;
- depth-3 accuracy: `+2.64 pp`;
- depth-4 accuracy: `+2.93 pp`;
- CE: `−0.3815` (improved).

The gain is positive in all four seeds. Rank16 adds `22,656` total and
estimated active parameters; router and circuit computation are unchanged.

## Decision

Rank16 cross-digit interaction is **validated as an unseen-range opt-in
direction across four seeds** and remains the leading quality/active-budget
candidate together with the four-seed full-range result in V0.230. The default
is unchanged. The next generalization stress test moves evaluation above the
trained operand interval (`train 0..63`, evaluate `64..95`) rather than adding
model capacity.

## Raw runs

Treatment:

- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction16_safeoffset_seed17_5000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction16_safeoffset_seed18_5000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction16_safeoffset_seed19_5000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction16_safeoffset_seed20_5000.json`

Control:

- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_safeoffset_seed17_5000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_safeoffset_seed18_5000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_safeoffset_seed19_5000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_safeoffset_seed20_5000.json`
