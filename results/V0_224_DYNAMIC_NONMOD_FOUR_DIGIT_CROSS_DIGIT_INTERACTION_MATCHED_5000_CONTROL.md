# V0.224 — matched 5000-step control for cross-digit interaction

**Date:** 2026-09-11  
**Status:** `VALIDATED OOD SIGNAL; FULL-RANGE CONTROL PENDING`

## Question

V0.223 showed a hard-accuracy improvement from rank-32 cross-digit interaction
at 5000 steps, but its CE could not yet be separated from a longer-training
effect. This audit compares the interaction treatment with a fresh
no-interaction control at the same steps, seeds, split, optimizer, and model
body.

## Protocol

- both arms use learned value encoding and the same polynomial2/Fourier state;
- four base-512 output digits, shared output rank `128`;
- treatment: `output_digit_interaction_rank=32`;
- control: `output_digit_interaction_rank=0`;
- training operands `0..31`, held-out operands `32..63`;
- training depths `1..2`, held-out depths `3..4`;
- safe target offset `33,554,432`;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- same factorized bank, active-8 route, and compact evaluator.

## Results

| Arm | Seed | Train accuracy | Unseen accuracy | Depth 3 | Depth 4 | CE |
|---|---:|---:|---:|---:|---:|---:|
| No interaction | 17 | 99.80% | 55.66% | 59.38% | 51.95% | 12.6858 |
| No interaction | 18 | 100.00% | 56.84% | 62.11% | 51.56% | 12.1238 |
| **No interaction mean** |  | **99.90%** | **56.25%** | **60.74%** | **51.76%** | **12.4048** |
| Interaction rank 32 | 17 | 100.00% | 59.38% | 62.50% | 56.25% | 12.2576 |
| Interaction rank 32 | 18 | 100.00% | 60.94% | 66.41% | 55.47% | 11.9096 |
| **Interaction mean** |  | **100.00%** | **60.16%** | **64.45%** | **55.86%** | **12.0836** |

Matched treatment minus control deltas:

- overall accuracy: `+3.91 pp`;
- depth-3 accuracy: `+3.71 pp`;
- depth-4 accuracy: `+4.10 pp`;
- CE: `−0.3212` (improved).

The interaction adds `45,312` total parameters and the same estimated active
budget increase (`2,170,808 → 2,216,120`), while router and selected circuit
computation are unchanged. The gain is present in both seeds and is not
explained by giving the treatment a longer budget.

## Decision

Cross-digit interaction rank 32 is **validated as a meaningful unseen-range
opt-in signal**. It supports the hypothesis that the previous ceiling is partly
caused by missing carry/cross-digit structure in the final readout, rather than
by router capacity alone.

It is not default yet. The required next gate is a matched full-range `0..63`
quality regression at 5000 steps, followed by rank/cost ablation if the
regression passes. This result still does not justify 700M/1B scaling.

## Raw runs

Treatment:

- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction32_safeoffset_seed17_5000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction32_safeoffset_seed18_5000.json`

Control:

- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_safeoffset_seed17_5000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_safeoffset_seed18_5000.json`
