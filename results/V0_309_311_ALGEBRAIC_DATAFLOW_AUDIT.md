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

## V0.320–V0.323 magnitude-coverage controls

These controls isolate the remaining high-value multiply failure without
changing the V0.316/V0.319 model body. V0.320 trains all depths with the
existing independent range mixture. V0.321 is a deliberately specialized
high-value-multiply-only control. V0.322 samples one value range per program
(`shared_per_program`). V0.323 keeps the broad training stream and adds a
25% targeted batch of homogeneous `80..95` multiply programs.

| Variant | Seed | Eval all | Eval d4 | Ordinary d4 multiply | High-value d4 multiply |
|---|---:|---:|---:|---:|---:|
| V0.320 all depths + independent mix | 17 | 98.7061% | 97.3633% | 49.2188% | 0.0000% |
| V0.321 high-value multiply only | 17 | diagnostic | diagnostic | — | 93.5547% |
| V0.322 all depths + shared range | 17 | 92.5049% | 91.9922% | 50.7813% | 7.8125% |
| V0.323 + targeted multiply 25% | 17 | 98.3887% | 96.8750% | 66.4063% | 80.4688% |
| V0.323 + targeted multiply 25% | 18 | 98.0957% | 96.4844% | 61.9141% | 83.3984% |
| V0.323 + targeted multiply 25% | 19 | 98.7305% | 98.1445% | 64.8438% | 81.8359% |
| V0.323 mean | 17/18/19 | 98.4050% | 97.1680% | 64.3880% | 81.9010% |
| V0.324 + targeted multiply 12.5% | 17 | 98.4863% | 96.4844% | 63.2813% | 72.6563% |

## V0.325 route-exploration control

V0.325 keeps the V0.323 25% targeted high-value multiply stream and raises
training-time `route_exploration_prob` from `0.05` to `0.15`. The purpose is
to test whether the V0.323 quality gain is paid for by excessive route
specialization. The inference path and active parameter budget are unchanged.

| Variant | Seed | Eval all | Eval d4 | Ordinary d4 multiply | High-value d4 multiply | Eval unique virtual circuits | Eval route entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0.323 targeted 25% | 17 | 98.3887% | 96.8750% | 66.4063% | 80.4688% | 569 | 5.2588 |
| V0.325 targeted 25% + route15 | 17 | 98.7305% | 96.7773% | 66.6016% | 79.1016% | 861 | 5.4292 |
| V0.326 targeted 25% + route10 | 17 | 98.6084% | 96.6797% | 66.0156% | 80.0781% | 669 | 5.6073 |

The control increases route coverage and factor-row coverage (33 to 44 unique
factor rows in the eval audit), while high-value d4 multiply falls by 1.3672
percentage points. This is evidence that exploration addresses the
specialization/coverage trade-off, but the tested probability is too costly
for the current quality objective. **V0.325 is retained as a diagnostic
coverage control, not as the default and not as a replacement for V0.323.**

V0.326 tests an intermediate exploration probability. It gives high-value d4
multiply `80.0781%`, ordinary d4 multiply `66.0156%`, eval d4 `96.6797%`,
and `669` unique virtual circuits with route entropy `5.6073`. It therefore
does not recover the V0.323 quality peak, even though coverage is higher than
V0.323. The route-exploration sweep is closed for now: V0.323 remains the
quality baseline, while V0.325/V0.326 remain coverage diagnostics.

## V0.327 matched 500M virtual-capacity control

V0.327 keeps the V0.323 task, representation, and training protocol while
increasing the virtual bank from `23,600` circuits / `154` factor rows to
`39,300` circuits / `199` factor rows. This raises total parameters from
`7,263,639` to `8,991,639`, but the hard active estimate remains
`1,964,480` because inference still selects eight circuits.

| Variant | Seed | Total params | Active params | Eval all | Eval d4 | Ordinary d4 multiply | High-value d4 multiply | Eval unique virtual circuits | Eval route entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0.323 300M targeted 25% | 17 | 7,263,639 | 1,964,480 | 98.3887% | 96.8750% | 66.4063% | 80.4688% | 569 | 5.2588 |
| V0.327 500M targeted 25% | 17 | 8,991,639 | 1,964,480 | 98.2422% | 96.4844% | 63.6719% | 81.6406% | 192 | 3.2928 |

The 500M control gives only a `+1.1718` percentage-point gain on high-value
d4 multiply, while ordinary d4 multiply drops `2.7344` points, broad eval d4
drops `0.3906` points, and route coverage collapses from 569 to 192 unique
virtual circuits. Therefore capacity alone does not solve the remaining
problem under this training budget; it makes specialization worse in this
seed. **V0.327 is rejected as a default upgrade.** The 300M V0.323 checkpoint
remains the quality baseline; larger scaling is deferred until the route and
capacity interaction is addressed or a matched longer-training experiment is
designed.

## V0.328 500M plus route10 control

V0.328 combines the 500M bank with the intermediate `route_exploration_prob`
of `0.10`, keeping the V0.323 targeted task mix unchanged. It improves the
500M ordinary d4 multiply result and route coverage relative to V0.327, but
still remains below the 300M baseline on broad quality and coverage.

| Variant | Seed | Eval all | Eval d4 | Ordinary d4 multiply | High-value d4 multiply | Eval unique virtual circuits | Eval route entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0.323 300M targeted 25% | 17 | 98.3887% | 96.8750% | 66.4063% | 80.4688% | 569 | 5.2588 |
| V0.327 500M targeted 25% | 17 | 98.2422% | 96.4844% | 63.6719% | 81.6406% | 192 | 3.2928 |
| V0.328 500M targeted 25% + route10 | 17 | 97.9492% | 95.6055% | 66.9922% | 81.6406% | 367 | 4.3160 |

V0.328 restores ordinary d4 multiply above V0.323 by `0.5859` points and
raises the 500M route coverage from 192 to 367 circuits, but high-value d4
does not improve over V0.327 and broad eval d4 falls by `1.2695` points versus
V0.323. **V0.328 is retained as a diagnostic interaction control, not as a
default.** The next 500M screen constrains the active factor capacity and
warms up routing, testing whether the larger virtual bank can be stabilized
without adding active inference cost.

V0.321 proves the architecture can learn the high-value multiply contract when
the task is isolated, but it destroys add/subtract generality and is not a
usable model. V0.322 shows that merely sharing the range per program is not
enough. V0.323 is the first balanced control that substantially raises
high-value multiply while preserving the broad task score: fixed depth-4 add
is `100%`, subtract is `99.6094%`, and high-value multiply is `80.4688%`.
Active parameters remain `1,964,480`; no extra circuit bank or router head was
added. The targeted stream does reduce route entropy and virtual-circuit
coverage, so a seed18 reproduction and route-specialization check remain
necessary before default adoption.

The seed18 reproduction confirms the direction: ordinary d4 multiply is
`61.9141%`, high-value d4 multiply is `83.3984%`, add is `100%`, and subtract
is `99.8047%`. Seed19 independently gives eval `98.7305%`, d4 `98.1445%`,
ordinary d4 multiply `64.8438%`, high-value d4 multiply `81.8359%`, add
`99.6094%`, and subtract `100%`. Across three seeds, high-value d4 multiply
averages `81.9010%` (80.4688/83.3984/81.8359), so the gain is reproducible.
The route audit still shows specialization (seed19 eval: 1,130 unique
virtual circuits, entropy `5.8698`), therefore this is a quality win with a
routing-coverage tradeoff, not evidence that the current 25% fraction is
optimal.

V0.324 halves the targeted fraction and preserves the broad eval score, but
high-value d4 multiply falls to `72.6563%` (seed17), below both V0.323 seeds.
This is still a positive lower-cost control, but it does not replace the
25% setting under the current 2,000-step budget.

**V0.320 RETAINED AS ALL-DEPTH CONTROL; V0.321 REJECTED AS A SPECIALIZED
MODEL; V0.322 REJECTED AS INSUFFICIENT; V0.323 RETAINED AS THE LEADING
BALANCED OPT-IN.** Three seeds now pass the basic quality reproduction gate,
but route coverage remains a live risk. The main diagnosis is data/task
coverage plus routing specialization around high-magnitude multiplication,
not a simple capacity shortage. Before 700M/1B scaling, run a route-coverage
control and keep the three-seed V0.323 mean as the acceptance baseline. V0.325
is the route-coverage reference for that follow-up.

Artifacts: V0.320–V0.325 configs, run reports, the `--include-trained-depths`
benchmark control, and operationwise JSONs.
