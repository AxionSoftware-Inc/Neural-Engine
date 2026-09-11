# V0.222 — four-digit cross-digit interaction on unseen operand range

**Date:** 2026-09-11  
**Status:** `PROMISING OPT-IN; NOT DEFAULT`

## Question

V0.220's four-digit codec trained cleanly on operands `0..31` but transferred
poorly to unseen operands `32..63`. V0.221 showed that replacing the learned
value encoder with fixed Fourier features made the result worse. This test
therefore targets the other likely bottleneck: independent digit heads may not
represent carry and cross-digit dependencies at the final readout.

The existing low-rank `output_digit_interaction_rank=32` path was enabled. Each
digit head receives a low-rank context derived from the preceding digit head's
probability distribution. The router, circuit bank, recurrent state, algebraic
state packet, optimizer, and input encoder were unchanged.

## Protocol

- learned input value encoder;
- four base-512 output digits, shared output rank `128`;
- cross-digit interaction rank `32`;
- training operands `0..31`, held-out operands `32..63`;
- training depths `1..2`, held-out depths `3..4`;
- safe target offset `33,554,432`;
- 3000 steps, batch `128`, seeds `17` and `18`;
- same factorized bank, active-8 route and compact evaluator as V0.220.

## Results

| Seed | Train accuracy | Unseen-range accuracy | Depth 3 | Depth 4 | Unseen-range CE | Total params | Active estimate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 98.83% | 53.52% | 56.25% | 50.78% | 10.2516 | 7,515,279 | 2,216,120 |
| 18 | 100.00% | 58.40% | 66.41% | 50.39% | 9.6239 | 7,515,279 | 2,216,120 |
| **Mean** | **99.41%** | **55.96%** | **61.33%** | **50.59%** | **9.9377** | **7,515,279** | **2,216,120** |

Against the matched V0.220 learned-encoder/no-interaction control:

- overall accuracy: `51.66% → 55.96%` (`+4.30 pp`);
- depth-3 accuracy: `55.47% → 61.33%` (`+5.86 pp`);
- depth-4 accuracy: `47.85% → 50.59%` (`+2.73 pp`);
- CE: `10.1647 → 9.9377` (`−0.2269`).

The interaction adds `45,312` total parameters and the same estimated active
budget increase, while selected circuit computation is unchanged. Both seeds
improve overall accuracy, although seed variance remains visible.

## Decision

This is a meaningful positive signal for the hypothesis that the unseen-range
ceiling is partly a cross-digit/carry readout problem. The interaction path is
**retained as an opt-in candidate**, but not made the default yet: the test is
only 3000 steps and uses the unseen-range split. A 5000-step two-seed
continuation and a full-range `0..63` quality regression control are required
before adoption.

This result does not justify 700M/1B scaling. It also does not prove that the
router is the primary bottleneck; the same router and circuit bank improved
when the final digit interface was made conditional.

## Reproduction

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction32_safeoffset.yaml --steps 3000 --seed 17 --run-id nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction32_safeoffset_seed17_3000 --checkpoint results/checkpoints/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction32_safeoffset_seed17_3000.pt --output results/runs --examples-per-depth 256 --log-every 500 --heldout-depths --train-value-min 0 --train-value-max 31 --eval-value-min 32 --eval-value-max 63
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction32_safeoffset.yaml --steps 3000 --seed 18 --run-id nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction32_safeoffset_seed18_3000 --checkpoint results/checkpoints/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction32_safeoffset_seed18_3000.pt --output results/runs --examples-per-depth 256 --log-every 500 --heldout-depths --train-value-min 0 --train-value-max 31 --eval-value-min 32 --eval-value-max 63
```

## Raw runs

- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction32_safeoffset_seed17_3000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction32_safeoffset_seed18_3000.json`
