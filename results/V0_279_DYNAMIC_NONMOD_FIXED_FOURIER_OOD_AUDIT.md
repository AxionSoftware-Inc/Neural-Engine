# V0.279 — Fixed-Fourier value representation on unseen values

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Question

V0.278 showed that a learned value curriculum improves held-out depth only
inside the trained value range and collapses on unseen values `64..95`. This
experiment replaces the learned value projection with the existing fixed
Fourier encoder to test whether an explicit periodic numeric representation
alone restores value extrapolation.

## Design

The 300M virtual-bank model, optimizer, schedule, and active path are kept
unchanged. Only `value_encoder_mode` changes from `learned` to
`fixed_fourier`. Training uses depths 1–2 and values
`0..7 → 0..31 → 0..63`; evaluation uses held-out depths 3–4 and unseen values
`64..95`. Each seed is evaluated with 1,024 identical examples per depth,
evaluation seed `4102`.

The fixed encoder is parameter-free. Total parameters therefore drop from
`7,483,463` to `7,478,087`; the active estimate remains approximately
`2,184,304`.

## Results

| Seed | OOD accuracy 64–95 | CE | Depth 3 | Depth 4 |
|---:|---:|---:|---:|---:|
| 17 | 0.586% | 41.5932 | 0.781% | 0.391% |
| 18 | 0.537% | 39.0932 | 0.781% | 0.293% |
| **Mean** | **0.562%** | **40.3432** | **0.781%** | **0.342%** |

For reference, the learned-value curriculum in V0.278 reached `2.661%` on
the same unseen range, while reaching `25.757%` on seen values `0..63`.

## Interpretation

Fixed Fourier coordinates do not solve the transfer problem. The failure is
not just an unstructured learned input lookup. The learned circuit/state
transition and the factorized categorical output codec still do not implement
an extrapolating numeric map for the unseen product digits. The extra route
coverage seen in these runs is therefore not sufficient evidence of useful
computation.

## Decision

**V0.279 REJECTED AS A RANGE-SAFE QUALITY SOLUTION.** It remains available as
a parameter-free diagnostic, but is not made default and does not justify
scaling the bank. The next test must add numeric/algebraic structure at the
state-to-output boundary (or an explicit range-safe digit/carry decoder),
while retaining the `64..95` OOD gate.

## Raw runs

- `results/runs/v0_279_ood64_95_fixed_fourier_paired_seed17_large.json`
- `results/runs/v0_279_ood64_95_fixed_fourier_paired_seed18_large.json`

Configuration:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_fixed_fourier_value_curriculum.yaml`.
