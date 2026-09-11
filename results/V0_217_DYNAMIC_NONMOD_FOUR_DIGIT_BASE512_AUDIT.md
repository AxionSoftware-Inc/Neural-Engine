# V0.217 — four-digit base-512 full-range codec

**Date:** 2026-09-11  
**Branch:** `exp/track-native-engine`  
**Status:** `LEADING FULL-RANGE OPT-IN / DEFAULT PENDING`

## Hypothesis

The failed three-digit full-range screen used 1024-way digit heads. The
failed two-digit screen showed that very large per-head class counts are hard
to optimize. This experiment keeps each head small by representing the same
`2^30` class space with four base-512 digits. The Fourier period is aligned to
the digit base. No router, circuit bank, recurrent state, or loss change is
made.

This is a representation-level test: it asks whether smaller local quotient /
remainder decisions improve the long-composition interface without adding
active circuit compute.

## Protocol

- full operand range: `0..63`;
- train program depths: `1..2`;
- held-out program depths: `3..4`;
- four output digits, base `512`;
- shared output projection rank `128`;
- Fourier base `512`;
- 3000 steps, batch `128`, same optimizer and target offset;
- seeds `17`, `18`, `19`, and fresh verification seed `20`;
- compact factorized evaluation;
- matched comparison: aligned three-digit base-1024 rank-128 codec.

## Results

| Run | Held-out accuracy | Depth 3 | Depth 4 | Held-out CE | Total params | Active estimate |
|---|---:|---:|---:|---:|---:|---:|
| Four-digit seed17 | 77.54% | 81.25% | 73.83% | 2.1662 | 7,469,967 | 2,170,808 |
| Four-digit seed18 | 81.64% | 88.28% | 75.00% | 2.3538 | 7,469,967 | 2,170,808 |
| Four-digit seed19 | 74.41% | 88.28% | 60.55% | 2.7523 | 7,469,967 | 2,170,808 |
| Four-digit seed20 | 78.52% | 85.55% | 71.48% | 2.3642 | 7,469,967 | 2,170,808 |
| **Four-seed mean** | **78.03%** | **85.84%** | **70.21%** | **2.4091** | — | — |
| Three-digit aligned mean | 67.97% | 77.54% | 58.40% | 3.4489 | 7,667,079 | 2,367,920 |

Across four seeds, the codec improves mean held-out accuracy by `+10.06 pp`
and depth-4 accuracy by `+11.81 pp` over the interaction-rank-0 three-digit
control. It does so with about `2.0%` fewer total parameters and `8.3%` fewer
estimated active parameters than the aligned three-digit control.

Training accuracy is `98.05%/99.02%/99.02%/96.48%`, so the improvement is not
caused by a train-collapse artifact. Seed19 has a much lower depth-4 score
despite strong depth-3 and train scores, so seed variance remains material.
The four-digit codec is now the leading full-range opt-in, but the default
change remains pending a longer matched run.

## Interpretation

The combined screens point to output-code granularity as a real bottleneck in
the full-range task:

1. three 1024-way heads: learnable but weak depth transfer;
2. two 32768-way heads: cannot learn the train split in budget;
3. four 512-way heads: strong train fit and much better depth transfer.

This does not prove that more digits always help. It shows that the model's
current state/circuit interface can support smaller local decisions much more
reliably than very large quotient heads. The next verification should test
whether the depth-4 result survives a longer matched run; only then should it
become the default full-range codec.

## Reproduction

```powershell
python -m pytest -q

python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128.yaml --steps 3000 --seed 17 --run-id nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed17_3000 --checkpoint results/checkpoints/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed17_3000.pt --output results/runs --examples-per-depth 256 --log-every 500 --heldout-depths --train-value-min 0 --train-value-max 63 --eval-value-min 0 --eval-value-max 63

python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128.yaml --steps 3000 --seed 18 --run-id nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed18_3000 --checkpoint results/checkpoints/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed18_3000.pt --output results/runs --examples-per-depth 256 --log-every 500 --heldout-depths --train-value-min 0 --train-value-max 63 --eval-value-min 0 --eval-value-max 63
```

Raw runs:

- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed18_3000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed19_3000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed20_3000.json`
