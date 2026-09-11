# V0.230 — four-seed rank-16 full-range validation

**Date:** 2026-09-11  
**Status:** `VALIDATED FULL-RANGE OPT-IN; OOD 4-SEED VALIDATION PENDING`

## Question

V0.229 identified rank16 cross-digit interaction as the best current
quality/cost point on two full-range seeds. This run adds seeds19/20 to test
whether the hard-quality advantage survives the observed seed variance.

## Protocol

- learned value encoder and polynomial2/Fourier algebraic state;
- four base-512 output digits, shared output rank `128`;
- `output_digit_interaction_rank=16`;
- training and evaluation operands `0..63`;
- training depths `1..2`, held-out depths `3..4`;
- safe target offset `33,554,432`;
- fresh 5000-step runs, batch `128`, seeds `17,18,19,20`;
- same factorized bank, active-8 route and compact evaluator.

## Results

| Seed | Train accuracy | Held-out accuracy | Depth 3 | Depth 4 | CE |
|---:|---:|---:|---:|---:|---:|
| 17 | 97.85% | 84.38% | 87.50% | 81.25% | 2.3826 |
| 18 | 99.41% | 83.40% | 89.84% | 76.95% | 2.7551 |
| 19 | 98.83% | 81.25% | 90.23% | 72.27% | 3.3528 |
| 20 | 99.41% | 84.57% | 91.41% | 77.73% | 2.2649 |
| **Mean** | **98.88%** | **83.40%** | **89.75%** | **77.05%** | **2.6888** |

The existing four-seed no-interaction full-range reference is `81.20%`
overall, `87.79%` depth-3, `74.61%` depth-4, and CE `2.6938`. The rank16
four-seed difference is therefore `+2.20 pp` overall, `+1.95 pp` depth-3,
`+2.44 pp` depth-4, and `−0.0049` CE. The reference uses the original target
offset while this validation uses the safe offset; V0.219's offset screen
showed negligible quality movement, but a fully paired four-seed safe-offset
control is not claimed here.

Rank16 adds `22,656` total and estimated active parameters over the
no-interaction reference, with no additional router or circuit execution.

## Decision

The rank16 hard-quality signal is **validated across four full-range seeds**
and remains the leading quality/cost opt-in candidate. It is not promoted to
default yet because the unseen-range gate currently has only seeds17/18; seeds
19/20 unseen-range validation is the next check. Capacity-only 700M/1B scaling
remains deferred.

## Raw runs

- `results/runs/nonmod_values0_63_four_digit_base512_rank128_interaction16_seed17_5000.json`
- `results/runs/nonmod_values0_63_four_digit_base512_rank128_interaction16_seed18_5000.json`
- `results/runs/nonmod_values0_63_four_digit_base512_rank128_interaction16_seed19_5000.json`
- `results/runs/nonmod_values0_63_four_digit_base512_rank128_interaction16_seed20_5000.json`
