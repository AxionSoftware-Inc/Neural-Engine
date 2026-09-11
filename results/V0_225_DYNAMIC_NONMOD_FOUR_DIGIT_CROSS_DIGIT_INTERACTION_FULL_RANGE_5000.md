# V0.225 — full-range regression for cross-digit interaction

**Date:** 2026-09-11  
**Status:** `HARD-QUALITY PASS; CE REGRESSION; OPT-IN`

## Question

V0.224 validated rank-32 cross-digit interaction on the unseen-range gate.
This test checks whether it damages the leading full-range-trained four-digit
codec when both training and evaluation use operands `0..63`.

## Protocol

- learned value encoder and polynomial2/Fourier algebraic state;
- four base-512 output digits, shared output rank `128`;
- treatment: `output_digit_interaction_rank=32`;
- training and evaluation operands `0..63`;
- training depths `1..2`, held-out depths `3..4`;
- safe target offset `33,554,432`;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- same factorized bank, active-8 route and compact evaluator as the baseline.

## Results

| Arm | Seed | Train accuracy | Held-out accuracy | Depth 3 | Depth 4 | CE |
|---|---:|---:|---:|---:|---:|---:|
| No interaction | 17 | 98.83% | 82.03% | 85.94% | 78.12% | 2.4796 |
| No interaction | 18 | 99.61% | 83.20% | 89.45% | 76.95% | 2.5717 |
| **No interaction mean** |  | **99.22%** | **82.62%** | **87.70%** | **77.54%** | **2.5257** |
| Interaction rank 32 | 17 | 98.05% | 83.01% | 85.94% | 80.08% | 2.4605 |
| Interaction rank 32 | 18 | 99.61% | 84.96% | 91.02% | 78.91% | 3.0164 |
| **Interaction mean** |  | **98.83%** | **83.98%** | **88.48%** | **79.49%** | **2.7384** |

Matched treatment minus control deltas:

- held-out accuracy: `+1.37 pp`;
- depth-3 accuracy: `+0.78 pp`;
- depth-4 accuracy: `+1.95 pp`;
- CE: `+0.2128` (worse).

The treatment adds `45,312` total and estimated active parameters
(`7,469,967/2,170,808 → 7,515,279/2,216,120`). Router and selected circuit
computation remain unchanged.

## Decision

The interaction passes the hard full-range regression gate and remains a
credible opt-in quality candidate. The CE regression, especially on seed18,
means it is not promoted to the default yet. The next experiment is a rank-16
ablation on the unseen-range gate to test whether a smaller interaction
interface preserves the gain and reduces calibration/cost overhead.

No 700M/1B scaling is justified by this result.

## Raw runs

Treatment:

- `results/runs/nonmod_values0_63_four_digit_base512_rank128_interaction32_seed17_5000.json`
- `results/runs/nonmod_values0_63_four_digit_base512_rank128_interaction32_seed18_5000.json`

Control:

- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed17_5000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed18_5000.json`
