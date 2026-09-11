# V0.228 — rank-8 cross-digit interaction full-range regression

**Date:** 2026-09-11  
**Status:** `REJECTED FOR DEFAULT; FULL-RANGE REGRESSION FAILURE`

## Question

V0.227 found that rank-8 interaction improves unseen-range accuracy at low
cost. This test checks whether that cheaper interface preserves the leading
full-range quality when training and evaluation both use operands `0..63`.

## Results

| Arm | Seed | Train accuracy | Held-out accuracy | Depth 3 | Depth 4 | CE |
|---|---:|---:|---:|---:|---:|---:|
| No interaction | 17 | 98.83% | 82.03% | 85.94% | 78.12% | 2.4796 |
| No interaction | 18 | 99.61% | 83.20% | 89.45% | 76.95% | 2.5717 |
| **No interaction mean** |  | **99.22%** | **82.62%** | **87.70%** | **77.54%** | **2.5257** |
| Interaction rank 8 | 17 | 97.85% | 77.34% | 81.25% | 73.44% | 2.5963 |
| Interaction rank 8 | 18 | 99.61% | 83.40% | 90.23% | 76.56% | 2.8052 |
| **Interaction rank 8 mean** |  | **98.73%** | **80.37%** | **85.74%** | **75.00%** | **2.7008** |

Matched treatment minus control deltas:

- held-out accuracy: `−2.25 pp`;
- depth-3 accuracy: `−1.95 pp`;
- depth-4 accuracy: `−2.54 pp`;
- CE: `+0.1751` (worse).

Rank8 adds only `11,328` total and estimated active parameters, but its
unseen-range benefit does not transfer to the full-range quality gate. Seed17
regresses substantially; seed18 is close to control but does not recover the
mean.

## Decision

Rank8 is **rejected for default adoption**. It remains a useful diagnostic and
low-cost OOD experiment, but cannot be the preferred production interaction
rank while full-range quality is the primary gate. Rank16 full-range
regression is still required to determine whether it is the practical
quality/cost compromise between rank8 and rank32.

## Raw runs

Treatment:

- `results/runs/nonmod_values0_63_four_digit_base512_rank128_interaction8_seed17_5000.json`
- `results/runs/nonmod_values0_63_four_digit_base512_rank128_interaction8_seed18_5000.json`

Control:

- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed17_5000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed18_5000.json`
