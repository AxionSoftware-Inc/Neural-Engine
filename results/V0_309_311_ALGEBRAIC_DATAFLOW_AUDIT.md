# V0.309–V0.311 — algebraic dataflow follow-up

**Date:** 2026-09-12  
**Status:** `V0.311 REJECTED FOR MAIN QUALITY ADOPTION`

## Question

V0.304 remains the leading learned opt-in branch, but its multiply path is
near chance. These three small screens test whether changing the interface
around the algebraic packet helps:

- **V0.309:** operation-conditioned rank-32 bilinear accumulator×operand
  write transition;
- **V0.310:** an additional multiply-only algebraic projection residual;
- **V0.311:** an additive `0.25` algebraic projection at the output bridge,
  while retaining the learned terminal decoder.

All variants keep the same 300M virtual factorized body, 8 active circuits,
8-digit base-16 compact evaluator, `0..95` training/evaluation range, depths
`1..2` train and `3..4` held out, and 2,000 steps unless noted. No default
model was changed.

## Overall results

| Variant | Seed(s) | Held-out all | Depth 3 | Depth 4 | CE | Total / active params |
|---|---:|---:|---:|---:|---:|---:|
| V0.304 reference | 17/18 | 38.5986% | 48.2422% | 28.9551% | 8.0427 | 7,236,759 / 1,937,600 |
| V0.309 bilinear rank32 | 17 | 39.0137% | 48.9258% | 29.1016% | 8.18177 | 7,348,503 / 2,049,344 |
| V0.310 multiply residual | 17 | 38.4277% | 48.9258% | 27.9297% | 8.86641 | 7,254,039 / 1,937,600 |
| V0.311 output bridge 0.25 | 17 | 40.6250% | 49.1211% | 32.1289% | 8.07340 | 7,236,759 / 1,937,600 |
| V0.311 output bridge 0.25 | 18 | 37.1094% | 47.1680% | 27.0508% | 7.90394 | 7,236,759 / 1,937,600 |
| V0.311 mean | 17/18 | 38.8672% | 48.1445% | 29.5898% | 7.98867 | 7,236,759 / 1,937,600 |

Relative to the two-seed V0.304 reference, V0.311 changes mean held-out
accuracy by only `+0.2686 pp`, depth-3 by `−0.0977 pp`, depth-4 by
`+0.6348 pp`, and CE by `−0.0540`. This is below the project’s meaningful
quality gate and is not stable evidence of a new capacity law.

## Operation-wise diagnostic

V0.311 fixed-operation results, averaged over seeds 17/18:

| Operation | Depth 3 | Depth 4 | High-value `80..95`, d3/d4 |
|---|---:|---:|---:|
| Add | 87.7930% | 63.4766% | 38.0859% / 23.6328% |
| Subtract | 99.7070% | 93.8477% | 100.0000% / 29.0039% |
| Multiply | 1.2695% | 1.6602% | 0.0000% / 0.0000% |

The V0.304 reference multiply was `1.5625%/1.6602%` at depths 3/4, so the
bridge did not improve multiplication and slightly worsened depth 3. The
high-value multiply failure is unchanged in both seeds. The operationwise
JSONs use the same 512 examples per case and the same fixed generator
protocol as prior screens.

## Decision

- **V0.309:** `REJECTED FOR ADOPTION`; rank-32 bilinear write is a small,
  single-seed change and does not address multiply.
- **V0.310:** `REJECTED`; a multiply-only projection residual is not enough and
  worsens CE/depth-4.
- **V0.311:** `REJECTED FOR MAIN QUALITY ADOPTION`; the two-seed gain is only
  `+0.2686 pp`, with no multiply or high-range signal.

V0.304 remains the leading prior-free learned opt-in branch. These screens
strengthen the diagnosis that the missing piece is a persistent learned
post-operation value/carry representation or output interface, not merely a
larger projection, a bilinear rank, or an additive output hint. No 700M/1B
scale-up follows from these results.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_operation_conditioned_product_algebraic_fourier_bilinear_write_rank32.yaml`
- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_operation_conditioned_product_algebraic_fourier_multiply_residual.yaml`
- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_operation_conditioned_product_algebraic_fourier_output_bridge025.yaml`
- `results/runs/v0_309_bilinear_write_rank32_seed17_2000.json`
- `results/runs/v0_310_multiply_algebraic_residual_seed17_2000.json`
- `results/runs/v0_311_output_bridge025_seed17_2000.json`
- `results/runs/v0_311_output_bridge025_seed18_2000.json`
- `results/operationwise_fixed_checkpoint_eval_v0_309_seed17.json`
- `results/operationwise_fixed_checkpoint_eval_v0_310_seed17.json`
- `results/operationwise_fixed_checkpoint_eval_v0_311_seed17.json`
- `results/operationwise_fixed_checkpoint_eval_v0_311_seed18.json`
- `results/operationwise_high_value_checkpoint_eval_v0_309_seed17.json`
- `results/operationwise_high_value_checkpoint_eval_v0_310_seed17.json`
- `results/operationwise_high_value_checkpoint_eval_v0_311_seed17.json`
- `results/operationwise_high_value_checkpoint_eval_v0_311_seed18.json`

## V0.312 typed-digit write follow-up

V0.312 adds the existing learned typed-digit carry state to the recurrent
write boundary only for multiply (`typed_digit_write_scale=0.25`,
`typed_digit_write_multiply_only=true`). Seed17, 2,000 steps, produced:

| Metric | V0.304 seed17 | V0.312 seed17 | Delta |
|---|---:|---:|---:|
| Held-out all | 39.5020% | 39.4531% | −0.0488 pp |
| Depth 3 | 49.2188% | 48.5352% | −0.6836 pp |
| Depth 4 | 29.7852% | 30.3711% | +0.5859 pp |
| CE | 8.0427 | 9.08746 | +1.0448 |

Fixed multiply was `2.3438%/1.7578%` at depths 3/4 and high-value
`80..95` multiply was `0%/0%`. **V0.312 REJECTED.** The typed register is
useful as a query-side diagnostic, but an additive multiply-only write hint
does not create a persistent learned multiplication contract.

Artifact: `results/runs/v0_312_typed_write025_multiply_only_seed17_2000.json`
and its paired operationwise JSONs.
