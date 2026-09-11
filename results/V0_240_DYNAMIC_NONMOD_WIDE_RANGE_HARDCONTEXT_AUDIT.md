# V0.240 — hard digit context on clean wide-support arithmetic

**Date:** 2026-09-11  
**Status:** `VALIDATED AS SMALL OPT-IN SIGNAL; DEFAULT UNCHANGED`

## Question

V0.235 showed that straight-through hard digit context helps subtract on the
confounded above-range screen but does not solve multiply. V0.239 removes the
product-range OOD confounder by training and evaluating operands `0..95` while
holding out only depths; multiply becomes non-zero but depth-4 remains weak.
This screen asks whether hard cross-digit context provides a real carry/codec
benefit on that clean protocol.

## Protocol

- rank-16 cross-digit interaction with `straight_through_hard` context;
- matched base rank-128 no-interaction control from V0.239;
- learned value encoder with `value_encoder_modulus=128`;
- polynomial2/Fourier algebraic state;
- training and evaluation operands `0..95`;
- train depths `1..2`, held-out depths `3..4`;
- target offset `134,217,728`, compact factorized evaluator;
- fresh 5000-step runs, batch `128`, seeds `17` and `18`;
- primary readout: operation-wise multiply and depth-4 accuracy; add/subtract
  are regression controls.

This is a clean depth-composition test, not an above-range value test and not
evidence for 700M/1B scaling.

## Results

The two-seed matched comparison against the V0.239 no-interaction control is:

| Metric | V0.239 control | V0.240 hard context | Delta |
|---|---:|---:|---:|
| Held-out all-ops accuracy | 79.1992% | 79.6875% | +0.4883 pp |
| Depth-3 accuracy | 84.5703% | 84.7656% | +0.1953 pp |
| Depth-4 accuracy | 73.8281% | 74.6094% | +0.7813 pp |
| Held-out CE | 3.554884 | 3.312335 | -0.242549 |

The task-wise held-out diagnosis gives the more important result:

| Operation | Control | Hard context | Delta |
|---|---:|---:|---:|
| Add | 100.00% | 100.00% | +0.00 pp |
| Subtract | 91.9922% | 99.7070% | +7.7148 pp |
| Multiply | 15.4297% | 15.4297% | +0.00 pp |
| All operations | 82.2266% | 82.7148% | +0.4883 pp |

For multiply, depth-3 is `23.2422% → 23.0469%` (`-0.1953 pp`) and
depth-4 is `7.6172% → 7.8125%` (`+0.1953 pp`). Thus the apparent aggregate
gain is entirely subtract-side; the deep multiply ceiling is unchanged.
The task-wise CE moves `2.744670 → 2.682122`, but this is also driven by the
large subtract improvement, not by a multiply solution.

Both runs used the same `7,500,743` total parameters and approximately
`2,201,584` active parameters. No capacity increase was used.

## Decision

Hard digit context is retained as an opt-in codec feature because it is a
repeatable subtract improvement on the clean wide-support protocol. It is not
made the default and is not evidence for 700M/1B scaling. The main P-003
failure is now localized more sharply: a multiply-specific recurrent
state/dataflow or transition is missing. The next experiment should change
that transition while keeping the wide-support/depth-holdout protocol and
the no-interaction control; another router or raw-capacity sweep is deferred.

Reproduction outputs:

- `results/runs/nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_value128_seed17_5000.json`
- `results/runs/nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_value128_seed18_5000.json`
- `results/runs/v0_240_wide_range_hardcontext_taskwise_seed17.json`
- `results/runs/v0_240_wide_range_hardcontext_taskwise_seed18.json`
