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

## V0.313 learned cross-digit pair table

V0.313 replaces the elementwise/soft-numeric multiply feature with learned
vector-valued tables for every contributing typed-digit pair and output slot.
The tables are evaluated against soft digit distributions; no exact
arithmetic lookup is used. Seed17, 2,000 steps, produced:

| Metric | V0.304 seed17 | V0.313 seed17 | Delta |
|---|---:|---:|---:|
| Held-out all | 39.5020% | 38.4766% | −1.0254 pp |
| Depth 3 | 49.2188% | 48.0469% | −1.1719 pp |
| Depth 4 | 29.7852% | 28.9062% | −0.8789 pp |
| CE | 8.0427 | 9.19522 | +1.1525 |

Total parameters rose to `7,392,375` and estimated active parameters to
`2,093,216`; training time rose to `496.56 s`. Fixed multiply was only
`1.7578%/0.9766%` at depths 3/4, and high-value multiply stayed `0%/0%`.
**V0.313 REJECTED.** A richer cross-digit pair interaction alone does not
repair the learned terminal value-to-digit path and adds avoidable inference
cost.

Artifact: `results/runs/v0_313_typed_pair_table_seed17_2000.json` and its
paired operationwise JSONs.

## V0.314–V0.316 precision and radix coverage

### V0.314 — multiply-only typed terminal readout

Switching the terminal output to the learned typed register only for multiply
damaged the shared training interface. Seed17 held-out accuracy was `28.2227%`
(`33.2031%/23.2422%` at depths 3/4) with CE `14.76864`; fixed and high-value
multiply were both `0%/0%`. **V0.314 REJECTED.** The typed register cannot yet
replace the learned terminal codec, even for one operation.

### V0.315 — double-precision algebraic register

The compact algebraic state was stored in float64 until Fourier feature
construction, avoiding low-order value loss from float32 normalization. Seed17
held-out accuracy was `41.0156%` (`50.3906%/31.6406%`) with CE `7.61083`.
Ordinary fixed multiply remained `1.9531%/1.5625%`; high-value multiply was
`0%/0%`. **V0.315 REJECTED AS A MULTIPLY FIX.** Precision affects aggregate
fit, but is not the main deep-multiply bottleneck.

### V0.316 — full radix-Fourier ladder

The algebraic feature packet was expanded from periods `16, 256, 2^30` to
all base-16 periods `16^1 ... 16^8`, while retaining float64 state storage.
This is a parameter-free feature coverage change; the learned projection and
sparse circuit body remain the same.

| Metric | Seed17 | Seed18 | Mean |
|---|---:|---:|---:|
| Held-out all | 87.9883% | 88.9160% | 88.4521% |
| Depth 3 | 93.0664% | 93.6523% | 93.3594% |
| Depth 4 | 82.9102% | 84.1797% | 83.5450% |
| CE | 2.12909 | 1.77765 | 1.95337 |

Against the V0.304 two-seed mean (`38.5986%`), V0.316 gains `+49.8536 pp`.
Fixed operationwise multiply rises to `26.7578%/6.8359%` for seed17 and
`28.3203%/8.9844%` for seed18, i.e. a mean `27.5391%/7.9102%` at depths 3/4.
Add and subtract are approximately `99–100%` in-range. However, homogeneous
high-value `80..95` multiply remains `0%/0%` for both seeds and depths.

**V0.316 RETAINED AS THE LEADING IN-RANGE LEARNED OPT-IN BRANCH, NOT YET A
GENERAL SOLUTION.** It is the first reproducible large quality jump from a
representation change, not from adding active circuits. The remaining issue
is high-magnitude/deep multiply coverage and extrapolation; scaling the bank
to 700M/1B is still premature until that gate is tested.

Artifacts include the V0.314/V0.315/V0.316 configs, run reports, and paired
operationwise JSONs.

## V0.317–V0.319 training-distribution follow-up

These screens keep the V0.316 full radix-Fourier ladder and test whether its
remaining deep/high-value gap is caused by the training distribution rather
than by the representation itself:

- **V0.317:** edge-mixture operand sampling, with 40% of examples from
  `80..95` and the remaining 60% from lower operand bands;
- **V0.318:** train depths `1..3` instead of only `1..2`;
- **V0.319:** both depth-3 training and edge-mixture sampling.

### Results

| Variant | Seed | Held-out all | Held-out d3 | Held-out d4 | CE |
|---|---:|---:|---:|---:|---:|
| V0.317 edge mix | 17 | 88.7207% | 92.9688% | 84.4727% | 1.81800 |
| V0.318 train d1–d3 | 17 | — | — | 93.3594% | 0.40132 |
| V0.319 train d1–d3 + edge mix | 17 | — | — | 95.1172% | 0.34604 |

V0.317 did not materially change the V0.316 in-range result and high-value
multiply remained `0%`. V0.318 and V0.319 show that exposing the model to
depth-3 compositions improves ordinary depth-4 transfer; V0.319 reaches
`95.1172%` on the aggregate depth-4 held-out set. This is a useful training
protocol signal, not proof of a universal capacity law: the current
operationwise high-value (`80..95`) multiply diagnostic is still `0%` for
V0.317–V0.319. Thus the remaining failure is specifically magnitude/depth
coverage or an extrapolating value contract, not simply insufficient total
parameter count or candidate routing.

**V0.317 RETAINED AS A DISTRIBUTION DIAGNOSTIC; V0.318/V0.319 RETAINED AS
PROMISING TRAINING-PROTOCOL OPT-IN RESULTS. None is made default, and no
700M/1B scale-up follows yet.** V0.319 should be reproduced on seed18 before
any adoption decision.

Artifacts: the V0.317–V0.319 configs, run reports, and operationwise JSONs.
