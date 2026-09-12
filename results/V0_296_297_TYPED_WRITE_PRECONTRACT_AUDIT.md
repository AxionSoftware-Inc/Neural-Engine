# V0.296/V0.297 — Typed write residual (pre-contract-fix record)

**Date:** 2026-09-12  
**Status:** `SUPERSEDED BY V0.298 CONTRACT FIX`

V0.296 added a `0.1×` typed-state residual at the actual accumulator write
boundary, restricted to multiply. V0.297 repeated the same variant for the
full 5,000-step seed17 run. These runs used the old leading-digit target loss,
so they are preserved as historical diagnostics and are not treated as clean
quality evidence.

| Run | Steps | Overall held-out acc | Depth 3 | Depth 4 | Fixed multiply d3 | Fixed multiply d4 |
|---|---:|---:|---:|---:|---:|---:|
| V0.296 seed17 | 2,000 | 2.7344% | 3.1250% | 2.3438% | 3.3203% | 2.3438% |
| V0.297 seed17 | 5,000 | 19.4336% | 26.0742% | 12.7930% | 3.5156% | 3.9063% |

V0.297 did not improve multiply over V0.290's fixed `3.9063%/4.1992%` and
reduced subtract quality. The residual was therefore not retained as a
quality default. The later V0.298 contract correction supersedes the loss and
training comparison; the code flag remains an opt-in diagnostic.

## Artifacts

- `neural_engine/dynamic_register.py`
- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_typed_write_residual.yaml`
- `results/operationwise_fixed_checkpoint_eval_v0_296.json`
- `results/operationwise_fixed_checkpoint_eval_v0_297.json`
