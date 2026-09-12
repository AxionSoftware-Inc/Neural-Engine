# V0.282 — Geometric factorized digit output

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Question

The learned prior-free lane fails on unseen values, while the algebraic lane
still uses categorical digit heads whose class positions are not ordered by
the loss. V0.282 adds an opt-in geometric head: each digit predicts one
continuous scalar coordinate and receives Gaussian distance logits against
ordered digit positions. The goal is to test whether this gives unseen digit
values a usable extrapolation path.

## Design

The existing 300M algebraic `polynomial2_fourier` state lane is kept, with
training values `0..63`, held-out depths 3–4, and OOD evaluation values
`64..95`. The standard categorical output is replaced only at the final
factorized head. The geometric temperature is `16.0` to avoid the large
initial gradients produced by squared distances over 512 positions.

This reduces the model from `7,477,191` to `7,271,307` total parameters and
the estimated active budget from about `2.05M` to `1.97M`; the sparse route and
algebraic state are otherwise unchanged.

## Results

Seed 17, 5,000 steps:

| Metric | Result |
|---|---:|
| Train accuracy, depths 1–2 | 1.953% |
| OOD accuracy, values 64–95 | 0.000% |
| OOD depth 3 / depth 4 | 0.000% / 0.000% |
| OOD CE | 410.2556 |
| Final training loss | 5.4924 |

The smoke test with temperature `2.0` was numerically finite but began near
loss `20,661`; it was corrected to temperature `16.0` before the full run.
With the corrected temperature, optimization still failed to learn the
factorized target. The one-seed failure is already far below the existing
algebraic reference, so a second expensive seed was not warranted.

## Decision

**V0.282 REJECTED.** Ordered Gaussian digit logits are not compatible with the
current state and compact digit-sum loss under this initialization/scale. The
feature remains opt-in code with a unit test, but it is not a quality path and
does not justify scaling. The next work should use the already validated exact
integer overlay/codec as a separate quality candidate, while keeping the
prior-free learned path's failure clearly labeled.

## Raw run

- `results/runs/v0_282_ood64_95_algebraic_geometric_output_seed17_5000.json`
- `results/runs/v0_282_ood64_95_algebraic_geometric_output_paired_seed17_large.json`

Configuration:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_algebraic_geometric_output.yaml`.
