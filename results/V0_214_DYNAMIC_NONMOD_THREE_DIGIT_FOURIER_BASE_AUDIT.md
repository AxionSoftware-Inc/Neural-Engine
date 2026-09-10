# V0.214 Three-Digit Fourier-Base Audit

## Question

V0.213 showed that a range-safe three-digit codec covers the `2^30` class
space but loses quality. Because the output digits use base `1024`, this audit
checks whether the Fourier bridge also needs base `1024` periods instead of
the inherited base `512` periods.

## Protocol

The model is unchanged apart from `algebraic_state_fourier_base`: three-digit
base-1024 output, shared rank-128 projection, `num_classes=2^30`, operands
`0--63`, train depths `1--2`, held-out depths `3--4`, target offset
`1,048,576`, batch size `128`, and `3,000` CUDA steps on seeds `17` and `18`.

## Results

| variant | seed | held-out | depth 3 | depth 4 | total params | active estimate |
|---|---:|---:|---:|---:|---:|---:|
| 3-digit base512 Fourier | 17 | 60.16% | 69.53% | 50.78% | 7.67M | 2.37M |
| 3-digit base512 Fourier | 18 | 66.41% | 73.44% | 59.38% | 7.67M | 2.37M |
| **base512 mean** | — | **63.28%** | **71.48%** | **55.08%** | — | — |
| 3-digit base1024 Fourier | 17 | 67.58% | 75.00% | 60.16% | 7.67M | 2.37M |
| 3-digit base1024 Fourier | 18 | 68.36% | 80.08% | 56.64% | 7.67M | 2.37M |
| **base1024 mean** | — | **67.97%** | **77.54%** | **58.40%** | — | — |

Matching the Fourier base to the output base improves the three-digit mean by
`4.69` points overall and `3.32` points at depth 4, but remains far below the
leading two-digit rank-128 `0--31` reference (`85.06%` overall,
`81.05%` depth 4).

## Interpretation

Period alignment is a real secondary factor, but it does not solve the main
problem. Independent additive digit logits still lose the cross-digit/value
interactions needed for unseen-depth composition. The class-range problem is
therefore separated from the quality problem: base alignment helps, while
three independent heads remain insufficient.

## Decision

**RETAIN BASE1024 PERIODS AS THE BEST THREE-DIGIT DIAGNOSTIC; REJECT THE
THREE-DIGIT CODEC FOR QUALITY ADOPTION.** The next codec must include
cross-digit interaction or a learned hierarchical carry/value path. Do not
scale the bank or model first.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_63_factorized_algebraic_three_digit_base1024_fourier1024_rank128.yaml`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_three_digit_base1024_fourier1024_rank128_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_three_digit_base1024_fourier1024_rank128_seed18_3000.json`

