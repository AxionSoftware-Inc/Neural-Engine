# V0.236 — hard-context task-wise diagnostic

**Date:** 2026-09-11  
**Status:** `DIAGNOSTIC COMPLETE; MULTIPLY CEILING CONFIRMED`

## Question

V0.235 validates a small four-seed gain from straight-through hard digit
context on the corrected `0..63 → 64..95` depth-holdout screen. This diagnostic
does not train a new model. It checks whether that gain is concentrated in one
primitive operation or one value/depth region, which determines whether the
variant is a reusable codec improvement or merely a narrow distribution effect.

## Protocol

- checkpoints: V0.235 hard-context rank16 treatments, seeds `17`, `18`, `19`,
  and `20`;
- deterministic evaluator seed and identical generated examples for each
  checkpoint;
- operand ranges `0..63` and `64..95`;
- train/held-out depth splits `1..2` and `3..4`;
- all operations together plus `add`, `subtract`, and `multiply` separately;
- compact factorized evaluation, so no dense `8.6B`-class logit materialization;
- no parameter updates and no active-budget changes.

The primary readout is held-out depth-3/4 accuracy by operation on the
above-range `64..95` distribution. This is a localization diagnostic, not a
new adoption gate.

## Reproduction

The evaluator now reads `generator_modulus`/`modulus` and `target_offset` from
the checkpoint configuration and supports repeated `--value-range MIN MAX`.
Example:

```powershell
python diagnose_dynamic_generalization.py `
  --checkpoint results/checkpoints/nonmod_train0_63_eval64_95_four_digit_rank128_interaction16_hardcontext_value128_seed17_5000.pt `
  --output results/runs/v0_236_hardcontext_taskwise_seed17.json `
  --examples-per-depth 256 `
  --value-range 0 63 `
  --value-range 64 95
```

## Results

All four hard-context treatment checkpoints and four matched no-interaction
controls were evaluated on the above-range `64..95` held-out distribution.
Each operation row aggregates four seeds, depths 3/4, and 256 examples per
depth. The control and treatment use the same compact factorized evaluator.

| operation | arm | held-out | depth-3 | depth-4 | CE |
|---|---|---:|---:|---:|---:|
| add | no-interaction control | 100.00% | 100.00% | 100.00% | 0.013073 |
| add | hard-context rank16 | 100.00% | 100.00% | 100.00% | 0.008087 |
| subtract | no-interaction control | 57.5195% | 87.3047% | 27.7344% | 8.443558 |
| subtract | hard-context rank16 | 78.8086% | 97.9492% | 59.6680% | 4.167495 |
| multiply | no-interaction control | 0.00% | 0.00% | 0.00% | 37.819530 |
| multiply | hard-context rank16 | 0.00% | 0.00% | 0.00% | 39.053754 |

The hard-context deltas are therefore `0.00 pp` for add, `+21.2891 pp`
overall / `+10.6445 pp` depth-3 / `+31.9336 pp` depth-4 for subtract, and
`0.00 pp` at every multiply accuracy level. Multiply CE is slightly worse
(`+1.2342`) under hard context. The V0.235 aggregate gain is thus not a
general routing improvement: it is primarily a subtract/readout interaction
gain, while the multiply composition path remains completely unresolved.

## Decision

The diagnostic confirms that hard digit context is useful as an opt-in
subtract codec, but it is not the fix for the main composition ceiling. Keep
it opt-in and do not make it the default. Do not scale to 700M/1B before a
separate multiply-state experiment shows non-zero held-out multiply accuracy.
The next architectural test should target operation-specific multiply state
transition/dataflow, with add and subtract held as controls.
