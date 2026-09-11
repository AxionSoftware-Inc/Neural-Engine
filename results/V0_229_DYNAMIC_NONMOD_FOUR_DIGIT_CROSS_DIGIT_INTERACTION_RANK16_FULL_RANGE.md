# V0.229 — rank-16 cross-digit interaction full-range regression

**Date:** 2026-09-11  
**Status:** `PROMISING QUALITY/COST CANDIDATE; 4-SEED VALIDATION PENDING`

## Question

V0.228 rejected rank8 because its unseen-range gain came with a full-range
regression. V0.225 showed rank32 passes full-range hard quality but costs more
and has a larger CE regression. This test checks whether rank16 is the
practical middle point.

## Protocol

- learned value encoder and polynomial2/Fourier algebraic state;
- four base-512 output digits, shared output rank `128`;
- treatment: `output_digit_interaction_rank=16`;
- training and evaluation operands `0..63`;
- training depths `1..2`, held-out depths `3..4`;
- safe target offset `33,554,432`;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- same factorized bank, active-8 route and compact evaluator.

## Results

| Arm | Seed | Train accuracy | Held-out accuracy | Depth 3 | Depth 4 | CE |
|---|---:|---:|---:|---:|---:|---:|
| No interaction | 17 | 98.83% | 82.03% | 85.94% | 78.12% | 2.4796 |
| No interaction | 18 | 99.61% | 83.20% | 89.45% | 76.95% | 2.5717 |
| **No interaction mean** |  | **99.22%** | **82.62%** | **87.70%** | **77.54%** | **2.5257** |
| Interaction rank 16 | 17 | 97.85% | 84.38% | 87.50% | 81.25% | 2.3826 |
| Interaction rank 16 | 18 | 99.41% | 83.40% | 89.84% | 76.95% | 2.7551 |
| **Interaction rank 16 mean** |  | **98.83%** | **83.89%** | **88.67%** | **79.10%** | **2.5689** |

Matched treatment minus control deltas:

- held-out accuracy: `+1.27 pp`;
- depth-3 accuracy: `+0.98 pp`;
- depth-4 accuracy: `+1.56 pp`;
- CE: `+0.0432` (slightly worse).

Rank16 adds `22,656` total and estimated active parameters
(`7,469,967/2,170,808 → 7,492,623/2,193,464`). Rank32's two-seed hard means
were `83.98%` overall and `79.49%` depth-4, so rank16 is nearly as strong at
about half the added budget and with a much smaller CE penalty.

## Decision

Rank16 is the current **leading quality/active-budget opt-in candidate**. It
passes the two-seed full-range hard regression and the two-seed unseen-range
gate from V0.226. It is not default yet: seed19/20 full-range validation and
broader held-out checks are required before promotion.

Capacity-only 700M/1B scaling remains deferred; the gain comes from a small
structured output interface, not a larger circuit bank.

## Raw runs

- `results/runs/nonmod_values0_63_four_digit_base512_rank128_interaction16_seed17_5000.json`
- `results/runs/nonmod_values0_63_four_digit_base512_rank128_interaction16_seed18_5000.json`
- unseen-range runs are listed in `V0_226_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_RANK16_OOD.md`.
