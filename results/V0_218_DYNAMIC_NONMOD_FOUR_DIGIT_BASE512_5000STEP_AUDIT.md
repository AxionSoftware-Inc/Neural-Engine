# V0.218 — four-digit base-512 5000-step verification

**Date:** 2026-09-11  
**Status:** `LEADING FULL-RANGE CANDIDATE / ROBUSTNESS PENDING`

## Question

The four-digit base-512 codec was the leading full-range representation after
four 3000-step seeds, but seed19 had a noticeably weak depth-4 result. This
screen tests whether that variance is partly an optimization-budget problem.

These are fresh trainings from the same initialization protocol, not resumed
checkpoints. That keeps the comparison clean: 3000 and 5000 steps differ only
in training budget.

## Protocol

- same four-digit base-512 / Fourier-base-512 config as V0.217;
- values `0..63`, train depths `1..2`, held-out depths `3..4`;
- shared rank-128 output projection;
- batch `128`, target offset `1,048,576`, identical optimizer;
- 5000 steps, seeds `17`, `18`, `19`, and `20`;
- compact factorized evaluation;
- matched 3000-step controls use the same seeds and config.

## Results

| Steps | Seed | Held-out accuracy | Depth 3 | Depth 4 | Held-out CE | Train accuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 3000 | 19 | 74.41% | 88.28% | 60.55% | — | 99.02% |
| 5000 | 19 | 79.49% | 88.28% | 70.70% | 2.9670 | 98.63% |
| 3000 | 20 | 78.52% | 85.55% | 71.48% | — | 96.48% |
| 5000 | 20 | 80.08% | 87.50% | 72.66% | 2.7567 | 99.41% |
| 5000 | 17 | 82.03% | 85.94% | 78.13% | 2.4796 | 98.24% |
| 5000 | 18 | 83.20% | 89.45% | 76.95% | 2.5717 | 99.61% |
| **5000 four-seed mean** | — | **81.20%** | **87.79%** | **74.61%** | **2.6938** | **98.97%** |

Compared with the matched four-seed 3000-step screen, 5000 steps add
`+3.17 pp` mean held-out accuracy and `+4.39 pp` mean depth-4 accuracy.
Seed19 benefits most: `+5.08/+10.16 pp` overall/depth-4. Seed20 adds
`+1.56/+1.17 pp`, while seed17/18 reach `82.03%/78.13%` and
`83.20%/76.95%` at overall/depth-4.

Against the aligned three-digit 3000-step control (`67.97%` overall,
`58.40%` depth-4), the 5000-step four-digit mean is ahead by `+13.23 pp`
and `+16.21 pp`, respectively. The codec remains at `7,469,967` total and
`2,170,808` estimated active parameters.

## Decision

This strengthens the representation-granularity hypothesis and shows that
some of the apparent seed variance was insufficient optimization budget. The
four-digit base-512 codec is now the current leading full-range candidate with
a four-seed 5000-step result. It remains opt-in while a target-offset and
fresh-data robustness screen checks that the gain is not tied to one absolute
label layout; the default config is unchanged for now.

## Raw runs

- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed19_5000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed20_5000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed17_5000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_four_digit_base512_fourier512_rank128_seed18_5000.json`
