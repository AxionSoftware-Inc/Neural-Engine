# V0.213 Three-Digit Output Codec Audit

## Question

The `0--63` depth-4 task can produce targets above the old two-digit
`67M`-class range. A three-digit base-1024 codec covers the full `2^30` class
space with only small heads. Does this range-safe representation preserve the
quality of the leading two-digit rank-128 codec?

## Architecture

`FactorizedDigitOutput` now supports an opt-in `output_digit_count=3` mode.
For `num_classes=2^30` and base `1024`, the target is represented by three
1024-way digit heads. All heads share a low-rank projection from the recurrent
state. Compact training sums the three digit cross-entropies and compact
evaluation reconstructs the exact integer argmax without materializing the
`2^30` Cartesian logit matrix.

The recurrent state, algebraic packet, Fourier bridge, router, circuit bank,
and active circuit budget are unchanged.

## Protocol

Operands are `0--63`, ordinary non-modular arithmetic, train depths `1--2`,
held-out depths `3--4`, target offset `1,048,576`, batch size `128`, and
`3,000` CUDA steps. Two seeds (`17`, `18`) are used for each rank. The leading
two-digit comparison is the V0.210 base-512 rank-128 result on `0--31`.

## Results

| codec | seed | held-out | depth 3 | depth 4 | train accuracy | total params | active estimate |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3-digit, rank 128 | 17 | 60.16% | 69.53% | 50.78% | 95.70% | 7.67M | 2.37M |
| 3-digit, rank 128 | 18 | 66.41% | 73.44% | 59.38% | 96.29% | 7.67M | 2.37M |
| **3-digit rank 128 mean** | — | **63.28%** | **71.48%** | **55.08%** | **96.00%** | — | — |
| 3-digit, rank 256 | 17 | 66.41% | 72.27% | 60.55% | 95.51% | 8.11M | 2.81M |
| 3-digit, rank 256 | 18 | 65.82% | 75.00% | 56.64% | 96.09% | 8.11M | 2.81M |
| **3-digit rank 256 mean** | — | **65.63%** | **73.63%** | **58.59%** | **95.80%** | — | — |
| two-digit base512 rank128, `0--31` | — | **85.06%** | **89.06%** | **81.05%** | **99.32%** | 15.79M | 10.49M |

Rank 256 improves the three-digit mean by only `2.34` points over rank 128,
and remains `19.43` points below the leading quality reference. The three-digit
codec is very small, but the quality loss is large and consistent enough to
reject it as the current quality path.

## Interpretation

Covering the integer range is not sufficient. Independent additive digit
logits appear to make the unseen-depth composition problem harder than the
two-factor codec used by the leading `0--31` model. Increasing the shared
latent rank from `128` to `256` does not recover the lost interaction. This
does not invalidate the shared rank-128 value codec; it rejects this specific
three-digit decomposition for adoption under the current loss and state
interface.

The failed depth-4 range-63 two-digit run is also not quality evidence: its
classifier range was invalid and the target guard correctly stopped it. A
future full-range codec would need structured cross-digit interactions or a
learned hierarchical value code, not merely more independent digit heads.

## Decision

**REJECT THREE-DIGIT CODEC FOR QUALITY ADOPTION.** Keep the implementation as
opt-in research infrastructure and preserve the V0.210/V0.211 shared rank-128
two-digit codec as the leading Native Engine path. Do not scale the circuit
bank or model to 700M/1B to compensate. The next architecture experiment must
add cross-digit/value-code interactions while preserving sparse active paths.

## Artifacts

- `neural_engine/dynamic_register.py` (`output_digit_count`)
- `train_dynamic_composition.py` (general compact digit loss/eval)
- `tests/test_dynamic_register.py`
- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_63_factorized_algebraic_three_digit_base1024_rank128.yaml`
- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_63_factorized_algebraic_three_digit_base1024_rank256.yaml`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_three_digit_base1024_rank128_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_three_digit_base1024_rank128_seed18_3000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_three_digit_base1024_rank256_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_three_digit_base1024_rank256_seed18_3000.json`

