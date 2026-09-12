# V0.280 — Algebraic state with progressive value curriculum

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Question

V0.274's curriculum strongly improved the prior-free model inside `0..95`,
but V0.278 showed that it did not extrapolate from `0..63` to `64..95`.
The repository's strongest learned-readout lane is the compact
`polynomial2_fourier` algebraic state bridge. V0.280 tests whether the value
curriculum adds anything to that lane on the same unseen-value gate.

## Design

The 300M virtual-bank model uses the existing polynomial/Fourier algebraic
state packet, learned value input encoder, four base-512 output digits, and no
cross-digit interaction. The only new training change is:

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
3–4 and unseen values `64..95`. The paired evaluator uses 1,024 examples per
depth and seed `4102`.

## Results

| Seed | OOD accuracy | CE | Depth 3 | Depth 4 | Train accuracy |
|---:|---:|---:|---:|---:|---:|
| 17 | 53.955% | 10.0302 | 59.082% | 48.828% | 99.609% |
| 18 | 55.713% | 12.1499 | 59.473% | 51.953% | 98.828% |
| **Mean** | **54.834%** | **11.0901** | **59.277%** | **50.391%** | **99.219%** |

The matched four-seed V0.233 no-interaction algebraic reference reached
`57.129%` OOD accuracy, `64.844%` at depth 3, and `49.414%` at depth 4.
Therefore the curriculum changes the balance but lowers overall OOD quality
by `2.295 pp` and depth-3 quality by `5.566 pp`; depth 4 rises by `0.977 pp`.

The curriculum did make training almost perfectly fit depths 1–2 and widened
route/factor usage, but that did not translate into higher unseen-value
accuracy. This separates training ease from reusable generalization.

## Decision

**CURRICULUM IS REJECTED FOR THE ALGEBRAIC OOD LANE.** Keep V0.274 as an
in-range opt-in recipe for the prior-free diagnostic, but do not combine it
with the leading algebraic state by default. The best current OOD quality
reference remains the existing V0.233/V0.240-family algebraic lane; its exact
integer overlay is tracked separately and is not evidence of a fully learned
sparse circuit.

No capacity increase follows. The next focused experiment is a low-rank
cross-digit interaction on this algebraic OOD protocol, because prior screens
showed a small positive signal there; it must beat the no-interaction
reference on both depth 3 and depth 4 without CE regression.

## Raw runs

- `results/runs/v0_280_ood64_95_algebraic_curriculum_paired_seed17_large.json`
- `results/runs/v0_280_ood64_95_algebraic_curriculum_paired_seed18_large.json`

Configuration:
`configs/ne_dynamic_300m_nonmod_train0_63_eval64_95_four_digit_base512_rank128_algebraic_curriculum.yaml`.
