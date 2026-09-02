# V0.43 OOD generalization and 700M parent-growth screen

Status: **700M growth is feasible and strong on supported/structured
composition; unseen value-range generalization remains unsolved**

## Scope

V0.42 showed that copying a trained 300M factorized parent into a 500M target
converts added capacity into useful quality. This milestone asks two separate
questions:

1. Does the result reproduce with a second 500M seed?
2. Is the high score genuine composition generalization, or does the model need
   to see every value token during training?

The experiments keep the non-Transformer typed-register graph and the same
factorized sparse bank. Checkpoints are local and ignored by Git.

## Second-seed 500M parent-growth

The seed-18 300M factorized model was expanded into a seed-18 500M target and
trained for 10,000 composition steps. The deterministic full `64^3` grid over
all nine ordered operation pairs produced:

| Model | Full all-pairs |
|---|---:|
| 500M growth, seed 17 | 99.6569% |
| 500M growth, seed 18 | **99.6675%** |
| Mean | **99.6622%** |

The 0.0106-point spread is much smaller than the earlier 100M/300M scratch
variance. The parent-growth result is therefore reproducible on this control.
The active estimate remains 1,792,305 parameters out of 18,973,217.

## Range-shift OOD audit

To test a genuinely unseen input region, new models were trained only on
operand values `0–31` and evaluated on `32–63`. All nine operator pairs remain
visible during training. The deterministic high-range grid contains
`32^3 = 32,768` examples per pair.

| Model | Training range accuracy | Unseen range accuracy |
|---|---:|---:|
| 300M global factorized | 99.74% | 27.35% |
| 500M growth global factorized | 99.85% | **28.57%** |
| 300M typed-only router | 99.61% | 28.14% |

The typed-only router removes value information from the route decision and
improves the range score by less than one point, so routing alone is not the
main rescue. More capacity also does not solve the unseen-token range shift.
This is a real negative result, but it is a harder protocol than the normal
all-range benchmark: the model never receives the high-half input tokens.

## Structured combination holdout

The range split is intentionally severe, so a second OOD protocol holds out
25% of operand triples within every operator pair while keeping all values
`0–63` visible during training. A deterministic FNV-style task-aware hash
assigns each triple to the train or held-out quarter.

| Model | Train combination split | Full held-out combination grid |
|---|---|---:|
| 300M factorized scratch | 75% of triples | 91.49% |
| **500M factorized parent-growth** | **75% of triples** | **99.68%** |

The 500M result shows that parent-growth can improve structured composition
generalization when the representation has already seen the value vocabulary.
It does not contradict the range-shift result; the two protocols test
different failure modes.

## 700M parent-growth feasibility

The 700M target uses 55,000 virtual route rows and 235 reusable factor rows,
for 25,785,153 physical parameters. It is initialized from the trained 500M
parent with `grow_factorized_capacity.py`; the new rows are initialized by the
target model while the learned parent prefix is copied.

The 3,000-step feasibility screen reached 99.8161% on the deterministic full
all-pairs grid. A clean 10,000-step run from the same growth initializer
reached:

| Metric | Result |
|---|---:|
| Full `64^3` all-pairs | **99.6627%** |
| Random all-pairs evaluation | 99.7396% |
| Active parameter estimate | 1,792,305 |
| Active fraction | 6.95% |

Per-pair full-grid accuracy stayed between 99.47% and 99.79%. The longer run
does not show a capacity-induced regression. A direct attempt to load the 500M
checkpoint into the larger model was correctly rejected by tensor shape checks;
the growth initializer is the required route-compatible transfer.

## Decision

**GO:** keep factorized parent-growth as the supported scaling recipe through
700M. It is strong on the all-range control, reproducible at 500M, and strong
on structured held-out combinations.

**NO-GO:** do not claim that capacity alone solves generalization to unseen
input ranges. The range-OOD score remains about 29% even at 500M, and
typed-only routing does not materially change it.

The next architecture/training target is a representation or teacher-transfer
path, not another scratch capacity jump. A Qwen3-style transfer should be
implemented as teacher distillation or an activation-to-register adapter; raw
Transformer neuron copying is not semantically compatible with typed-register
circuits. Keep the current 700M checkpoint frozen as the quality control while
that separate experiment is developed.

## Reproduction

- Second-seed 500M: `configs/ne_typed_register_500m_factorized_all_seed18.yaml`.
- Range OOD: `configs/ne_typed_register_300m_factorized_ood_range.yaml` and
  `configs/ne_typed_register_500m_factorized_ood_range.yaml`.
- Typed-only OOD: `configs/ne_typed_register_300m_factorized_ood_typed_route.yaml`.
- Combination holdout: `configs/ne_typed_register_300m_factorized_ood_combo.yaml`
  and `configs/ne_typed_register_500m_factorized_ood_combo.yaml`.
- 700M target: `configs/ne_typed_register_700m_factorized_growth.yaml`.
- Run JSON files and checkpoints are under ignored `results/runs/` and
  `results/checkpoints/` paths.
