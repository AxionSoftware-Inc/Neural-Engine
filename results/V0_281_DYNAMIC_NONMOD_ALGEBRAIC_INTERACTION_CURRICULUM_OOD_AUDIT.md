# V0.281 — Algebraic OOD lane with interaction and curriculum

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Question

Earlier above-range screens found a small positive signal from rank-16
cross-digit interaction, while V0.280 showed that progressive value curriculum
did not improve the algebraic lane. V0.281 combines both controls to test
whether they complement one another on unseen values `64..95`.

## Design

The 300M virtual-bank model keeps the existing `polynomial2_fourier` algebraic
state, learned value encoder, four base-512 output digits, active-8 sparse
route, and 5,000-step protocol. It adds only
`output_digit_interaction_rank=16` and the same value curriculum:

```yaml
value_curriculum:
  - until_step: 1000
    value_min: 0
    value_max: 7
  - until_step: 2500
    value_min: 0
    value_max: 31
  - until_step: 5000
    value_min: 0
    value_max: 63
```

Training uses depths 1–2 and values `0..63`; evaluation uses held-out depths
3–4 and unseen values `64..95`. Paired evaluation uses 1,024 examples per
depth and seed `4102`.

## Results

| Seed | OOD accuracy | CE | Depth 3 | Depth 4 | Train accuracy |
|---:|---:|---:|---:|---:|---:|
| 17 | 55.811% | 11.7786 | 59.668% | 51.953% | 99.219% |
| 18 | 57.324% | 12.1447 | 61.230% | 53.418% | 98.828% |
| **Mean** | **56.567%** | **11.9616** | **60.449%** | **52.686%** | **99.023%** |

The matched V0.233 rank-16 interaction reference without curriculum reached
`58.105%` mean OOD accuracy, `65.820%` at depth 3, and `50.391%` at depth 4.
V0.281 is therefore `−1.538 pp` overall and `−5.371 pp` at depth 3, while
depth 4 improves `+2.295 pp`. The mixed depth result is not sufficient for
adoption, and CE does not provide a compensating improvement.

## Interpretation

The two mechanisms do not add constructively under this budget. The
curriculum makes the training path fit depths 1–2 easily and increases route
coverage, but the interaction head does not turn that into better value
extrapolation. The remaining OOD ceiling is therefore not solved by stacking
training schedule and output interaction changes.

## Decision

**V0.281 REJECTED FOR QUALITY ADOPTION.** Keep rank-16 interaction as an
opt-in diagnostic only, and keep curriculum separate as an in-range protocol.
Do not scale the bank to 700M/1B from this result. The strongest quality
reference remains the existing algebraic/integer codec lane, which must be
reported separately from a fully learned prior-free circuit result.

The next experiment should reduce the exact numeric readout's dependence on a
large learned categorical overlay: test a compact range-calibrated integer
codec or continuous digit-coordinate decoder while preserving the learned
router/state path and the same unseen-value gate.

## Raw runs

- `results/runs/v0_281_ood64_95_algebraic_interaction16_curriculum_paired_seed17_large.json`
- `results/runs/v0_281_ood64_95_algebraic_interaction16_curriculum_paired_seed18_large.json`

Configuration:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_algebraic_interaction16_curriculum.yaml`.
