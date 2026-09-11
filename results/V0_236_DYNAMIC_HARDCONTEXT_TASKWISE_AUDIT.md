# V0.236 — hard-context task-wise diagnostic

**Date:** 2026-09-11  
**Status:** `RUN CONFIGURED; RESULTS PENDING`

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

Pending completion.
