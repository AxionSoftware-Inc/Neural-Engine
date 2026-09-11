# V0.223 — 5000-step cross-digit interaction on unseen operand range

**Date:** 2026-09-11  
**Status:** `MIXED POSITIVE; MATCHED CONTROL REQUIRED`

## Question

V0.222 found a promising `+4.30 pp` unseen-range gain from rank-32
cross-digit output interaction at 3000 steps. This continuation checks whether
the hard-accuracy gain survives a longer budget and whether its calibration
remains healthy.

## Protocol

- same config and unseen-range split as V0.222;
- learned value encoder, polynomial2/Fourier algebraic state;
- four base-512 output digits, shared rank `128`;
- `output_digit_interaction_rank=32`;
- training operands `0..31`, held-out operands `32..63`;
- training depths `1..2`, held-out depths `3..4`;
- safe target offset `33,554,432`;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- same factorized bank, active-8 route and compact evaluator.

## Results

| Seed | Train accuracy | Unseen-range accuracy | Depth 3 | Depth 4 | Unseen-range CE | Total params | Active estimate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.00% | 59.38% | 62.50% | 56.25% | 12.2576 | 7,515,279 | 2,216,120 |
| 18 | 100.00% | 60.94% | 66.41% | 55.47% | 11.9096 | 7,515,279 | 2,216,120 |
| **Mean** | **100.00%** | **60.16%** | **64.45%** | **55.86%** | **12.0836** | **7,515,279** | **2,216,120** |

Relative to the same interaction variant at 3000 steps (V0.222), hard accuracy
improves by `+4.20 pp` overall, `+3.13 pp` at depth 3, and `+5.27 pp` at
depth 4. However, mean CE rises from `9.9377` to `12.0836`. The 5000-step
no-interaction control has not yet been run, so the CE change cannot be
attributed to interaction rather than longer optimization.

## Decision

The hard-accuracy signal is retained as **promising opt-in**, but adoption is
blocked pending two matched controls:

1. 5000-step learned-encoder/no-interaction unseen-range control;
2. 5000-step interaction full-range `0..63` regression control.

The variant remains opt-in and the default is unchanged. This result still
does not justify 700M/1B scaling; it points to the output carry/readout
interface as a higher-value target than raw capacity.

## Raw runs

- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction32_safeoffset_seed17_5000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_interaction32_safeoffset_seed18_5000.json`
