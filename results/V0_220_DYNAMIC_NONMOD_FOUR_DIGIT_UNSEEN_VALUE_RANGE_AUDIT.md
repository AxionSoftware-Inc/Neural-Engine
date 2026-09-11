# V0.220 — four-digit codec on unseen operand range

**Date:** 2026-09-11  
**Status:** `PARTIAL POSITIVE / OOD PROBLEM STILL OPEN`

## Question

The four-digit base-512 codec is strong when training and evaluation both use
values `0..63`. This test asks whether it also generalizes to a completely
unseen operand half: train only on `0..31`, then evaluate on `32..63`.

The first attempt used the normal `1,048,576` target offset and was correctly
stopped by the target-range guard: large unmodulated subtract/multiply chains
produced negative evaluation labels. That run is **invalid and has no quality
interpretation**. The valid rerun uses offset `33,554,432`, which covers the
observed target range while preserving the same four-digit codec and model
body.

## Valid protocol

- four base-512 output digits, shared rank `128`;
- values `0..31` for training;
- values `32..63` for evaluation;
- train depths `1..2`, held-out depths `3..4`;
- target offset `33,554,432` for a valid non-negative label range;
- 3000 steps, batch `128`, seeds `17` and `18`;
- same router, circuit bank, optimizer, and active budget as the leading
  full-range candidate.

## Results

| Seed | Train accuracy | Unseen-range accuracy | Depth 3 | Depth 4 | Unseen-range CE |
|---|---:|---:|---:|---:|---:|
| 17 | 98.83% | 52.73% | 55.08% | 50.39% | 9.8748 |
| 18 | 99.61% | 50.59% | 55.86% | 45.31% | 10.4547 |
| **Mean** | **99.22%** | **51.66%** | **55.47%** | **47.85%** | **10.1647** |

The model learns the seen operand range cleanly but loses roughly half of its
hard accuracy on the unseen range. This is a real generalization limitation,
not an optimization failure: train loss is near zero and both seeds reproduce
the drop.

## Decision

The four-digit codec remains the leading **full-range-trained** candidate, but
it is not a solution to unseen-value extrapolation. The OOD problem remains
active under P-003. The next research direction should be an explicit
value/carry algebraic contract or a carefully controlled value-range
curriculum; simply increasing capacity or changing the digit count is not
enough.

## Raw runs

- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_safeoffset_seed17_3000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_safeoffset_seed18_3000.json`
