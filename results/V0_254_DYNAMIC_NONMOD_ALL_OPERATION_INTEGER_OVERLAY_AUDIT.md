# V0.254 — all-operation frozen integer overlay

**Date:** 2026-09-11  
**Status:** **RETAINED AS STRONG OPT-IN NUMERIC READOUT**

## Question

V0.252 used the exact integer packet only for terminal multiply. V0.253 showed
that adding that packet to the shared recurrent query was harmful and did not
change multiply. This experiment keeps the body, router, circuit bank, and
V0.252 overlay initialization frozen, but lets a separate factorized digit
head read the exact integer packet for **all three operations** at the terminal
output.

This isolates whether the problem is the learned terminal numeric readout,
rather than routing or recurrent state capacity.

## Protocol

- initialization: V0.252 codec-calibrated frozen overlay, one checkpoint per
  seed
- seeds: `17`, `18`
- overlay training: `5,000` steps, batch `128`
- train values: `0..95`; held-out depths: `3..4`
- evaluation: `256` examples per depth, values `0..95`
- full model: `7,838,217` parameters
- trainable overlay: `337,474` parameters
- body/router/circuit bank: frozen
- only change from V0.252: integer output head is selected for `all`
  operations instead of `multiply_only`

## Matched held-out results

| Variant | All ops | Depth 3 | Depth 4 | Add | Subtract | Multiply | Mul d3 | Mul d4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0.240 base | 82.7148% | 86.5234% | 78.9063% | 100.0000% | 99.7070% | 15.4297% | 23.0469% | 7.8125% |
| V0.252 multiply-only | 84.1797% | 87.5000% | 80.8594% | 100.0000% | 99.7070% | 59.8633% | 80.0781% | 39.6484% |
| **V0.254 all-operation** | **97.8516%** | **98.4375%** | **97.2656%** | **100.0000%** | **100.0000%** | **71.8750%** | **95.8984%** | **47.8516%** |

The V0.254 aggregate and multiply values are means over seeds17/18. Seed-wise
aggregate was `98.0469% / 97.6562%`; seed-wise multiply was
`72.4609% / 71.2891%`. Add and subtract were `100%` in both seeds.

## Decision and interpretation

**Retained as a strong opt-in candidate; default remains unchanged.** The
all-operation head improves aggregate accuracy by `+13.6719 pp` over V0.252
and multiply by `+12.0117 pp`, without changing the frozen body or routing.

This is strong evidence that the dominant failure in this synthetic
non-modular benchmark is terminal numeric representation/readout. It is not
evidence that a larger circuit bank or a better router would solve the issue.
The exact integer register also contains the known arithmetic transition, so
this result must not be presented as general language-model quality or as a
proof that the learned circuits have learned arithmetic independently.

The next question is range coverage: V0.254's codec calibration still covered
only the old `±81,450,625` raw-value range. V0.255 expands that calibration to
the full legal output class range.

## Artifacts

- config: `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_integeroverlay_codec_calibration_w005_all.yaml`
- trainer: `train_integer_output_overlay.py`
- checkpoints: `results/checkpoints/v0_254_all_operation_integeroverlay_seed17_5000.pt`,
  `...seed18_5000.pt`
- taskwise reports: `results/runs/v0_254_all_operation_integeroverlay_taskwise_seed17.json`,
  `...seed18.json`
