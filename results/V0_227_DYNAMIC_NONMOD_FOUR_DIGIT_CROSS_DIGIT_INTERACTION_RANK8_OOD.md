# V0.227 — rank-8 cross-digit interaction cost ablation

**Date:** 2026-09-11  
**Status:** `PROMISING LOW-COST OPT-IN; FULL-RANGE CONTROL PENDING`

## Question

V0.226 showed that rank-16 cross-digit interaction preserves most of the
rank-32 unseen-range gain at half the added budget. This test reduces the
conditional digit context to rank 8.

## Protocol

- treatment: `output_digit_interaction_rank=8`;
- control: `output_digit_interaction_rank=0`;
- learned value encoder and polynomial2/Fourier algebraic state;
- four base-512 output digits, shared output rank `128`;
- training operands `0..31`, held-out operands `32..63`;
- training depths `1..2`, held-out depths `3..4`;
- safe target offset `33,554,432`;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- same factorized bank, active-8 route and compact evaluator.

## Results

| Arm | Seed | Train accuracy | Unseen accuracy | Depth 3 | Depth 4 | CE |
|---|---:|---:|---:|---:|---:|---:|
| No interaction | 17 | 99.80% | 55.66% | 59.38% | 51.95% | 12.6858 |
| No interaction | 18 | 100.00% | 56.84% | 62.11% | 51.56% | 12.1238 |
| **No interaction mean** |  | **99.90%** | **56.25%** | **60.74%** | **51.76%** | **12.4048** |
| Interaction rank 8 | 17 | 100.00% | 57.62% | 60.55% | 54.69% | 12.4164 |
| Interaction rank 8 | 18 | 100.00% | 59.96% | 62.89% | 54.69% | 12.1251 |
| **Interaction rank 8 mean** |  | **100.00%** | **58.79%** | **61.72%** | **54.69%** | **12.2707** |

Matched treatment minus control deltas:

- overall accuracy: `+2.54 pp`;
- depth-3 accuracy: `+0.98 pp`;
- depth-4 accuracy: `+2.93 pp`;
- CE: `−0.1341` (improved).

Rank8 adds `11,328` total and estimated active parameters
(`7,469,967/2,170,808 → 7,481,295/2,182,136`). Its mean overall accuracy is
only `0.10 pp` below rank16 and its depth-4 accuracy is `0.59 pp` below rank16,
while costing half as much extra budget.

## Decision

Rank8 is retained as a **promising low-cost opt-in**. The unseen-range gain is
still consistent across both seeds, but full-range `0..63` regression is
required before selecting it as the preferred interaction rank. Rank32 remains
the higher-quality reference; rank8 may be the better active-budget trade-off.

## Raw runs

- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction8_safeoffset_seed17_5000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction8_safeoffset_seed18_5000.json`
- controls are listed in `V0_224_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_MATCHED_5000_CONTROL.md`.
