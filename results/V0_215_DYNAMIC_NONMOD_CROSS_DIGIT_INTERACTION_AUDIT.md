# V0.215 — cross-digit interaction in the factorized output codec

**Date:** 2026-09-11  
**Branch:** `exp/track-native-engine`  
**Status:** `REJECTED FOR DEFAULT / RETAINED AS OPT-IN`

## Question

The aligned three-digit Fourier codec improved the three-digit range screen
over the smaller-base version, but remained well below the leading two-digit
codec. The hypothesis was that the three digit heads were too independent:
the prediction of a later digit should be able to use information from the
earlier digit distribution.

This experiment adds a soft, differentiable cross-digit context path. After a
digit head produces logits, its softmax distribution is multiplied by a
learned embedding table. A rank-32 projection of that expected embedding is
added to the shared classifier state before the next digit head. The circuit
bank, router, recurrent state, training data, loss, and inference budget are
otherwise unchanged.

The context is soft rather than teacher-forced, so evaluation and training use
the same predicted-distribution path.

## Protocol

- full target range: values `0..63`;
- train program depths: `1..2`;
- held-out program depths: `3..4`;
- `3` output digits, base `1024`;
- shared output projection rank `128`;
- cross-digit interaction rank `32`;
- `3000` steps, batch `128`, identical optimizer and schedule;
- seeds `17` and `18`;
- compact factorized evaluation and the same target offset `1,048,576`;
- comparison: the matched interaction-rank-0 aligned three-digit control.

## Results

| Run | Held-out accuracy | Depth 3 | Depth 4 | Held-out CE | Total params | Active estimate |
|---|---:|---:|---:|---:|---:|---:|
| Control seed17 | 67.58% | 75.00% | 60.16% | 3.3118 | 7,667,079 | 2,367,920 |
| Control seed18 | 68.36% | 80.08% | 56.64% | 3.5860 | 7,667,079 | 2,367,920 |
| Interaction seed17 | 69.14% | 75.39% | 62.89% | 3.2065 | 7,740,807 | 2,441,648 |
| Interaction seed18 | 68.55% | 80.47% | 56.64% | 3.8294 | 7,740,807 | 2,441,648 |
| **Two-seed mean control** | **67.97%** | **77.54%** | **58.40%** | **3.4489** | — | — |
| **Two-seed mean interaction** | **68.85%** | **77.93%** | **59.77%** | **3.5180** | — | — |

Relative to the matched control, mean hard accuracy improved only `+0.88 pp`,
depth-4 improved `+1.37 pp`, and depth-3 improved `+0.39 pp`. Mean CE
regressed by `+0.0691`, driven by seed18. The extra context path adds only
`73,728` total parameters and the same amount to the active estimate, but the
quality gate is not met and the effect is not stable enough to make default.

## Decision

This is a **small positive accuracy signal, not a capacity solution**. It does
not explain why the larger model fails to scale, and it does not close the gap
to the leading two-digit codec. The implementation remains opt-in for future
ablation; the default output codec and model behavior are unchanged.

The result also warns against using hard accuracy alone: the apparent accuracy
gain came with worse mean CE and no depth-4 gain on seed18. Any follow-up must
report both metrics and preserve the matched two-seed protocol.

## Reproduction

```powershell
python -m pytest -q

python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_63_factorized_algebraic_three_digit_base1024_fourier1024_rank128_interaction32.yaml --steps 3000 --seed 17 --run-id nonmod_depth4_values0_63_factorized_algebraic_three_digit_base1024_fourier1024_rank128_interaction32_seed17_3000 --checkpoint results/checkpoints/nonmod_depth4_values0_63_factorized_algebraic_three_digit_base1024_fourier1024_rank128_interaction32_seed17_3000.pt --output results/runs --examples-per-depth 256 --log-every 500 --heldout-depths --train-value-min 0 --train-value-max 63 --eval-value-min 0 --eval-value-max 63

python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_63_factorized_algebraic_three_digit_base1024_fourier1024_rank128_interaction32.yaml --steps 3000 --seed 18 --run-id nonmod_depth4_values0_63_factorized_algebraic_three_digit_base1024_fourier1024_rank128_interaction32_seed18_3000 --checkpoint results/checkpoints/nonmod_depth4_values0_63_factorized_algebraic_three_digit_base1024_fourier1024_rank128_interaction32_seed18_3000.pt --output results/runs --examples-per-depth 256 --log-every 500 --heldout-depths --train-value-min 0 --train-value-max 63 --eval-value-min 0 --eval-value-max 63
```

Raw run JSON files:

- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_three_digit_base1024_fourier1024_rank128_interaction32_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_three_digit_base1024_fourier1024_rank128_interaction32_seed18_3000.json`
