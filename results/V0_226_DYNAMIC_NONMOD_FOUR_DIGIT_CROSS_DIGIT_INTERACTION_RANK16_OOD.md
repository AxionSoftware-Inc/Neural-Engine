# V0.226 — rank-16 cross-digit interaction cost ablation

**Date:** 2026-09-11  
**Status:** `PROMISING LOWER-COST OPT-IN`

## Question

V0.224 validated rank-32 cross-digit output interaction on the unseen-range
gate. This test asks whether a rank-16 context is enough to retain a useful
carry/readout gain while reducing the additional active budget.

## Protocol

- treatment: `output_digit_interaction_rank=16`;
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
| Interaction rank 16 | 17 | 100.00% | 57.81% | 60.16% | 55.47% | 11.8653 |
| Interaction rank 16 | 18 | 100.00% | 59.96% | 64.84% | 55.08% | 11.7561 |
| **Interaction rank 16 mean** |  | **100.00%** | **58.89%** | **62.50%** | **55.27%** | **11.8107** |

Matched treatment minus control deltas:

- overall accuracy: `+2.64 pp`;
- depth-3 accuracy: `+1.76 pp`;
- depth-4 accuracy: `+3.52 pp`;
- CE: `−0.5941` (improved).

Rank16 adds `22,656` total and estimated active parameters
(`7,469,967/2,170,808 → 7,492,623/2,193,464`). Rank32's corresponding mean
gains were `+3.91/+4.10 pp` overall/depth-4, so rank16 gives a smaller but
still consistent hard-quality gain at roughly half the added budget.

## Decision

Rank16 is retained as a **promising lower-cost opt-in**. It is not default yet:
the next screen is rank8 on the same unseen-range gate, followed by a
full-range regression for the best cost/quality point.

## Raw runs

- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction16_safeoffset_seed17_5000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction16_safeoffset_seed18_5000.json`
- controls are listed in `V0_224_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_MATCHED_5000_CONTROL.md`.
