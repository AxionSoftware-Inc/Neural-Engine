# V0.306–V0.308 — multiply codec, 5k learned control, and operation-conditioned bridge

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## V0.306: frozen exact-integer multiply overlay

V0.304's learned body/router was frozen. Only a multiply-specific exact
integer output decoder, digit embeddings, and factorized digit head were
trained. A small random raw-value codec calibration (`weight=0.05`) was
included. This is a diagnostic overlay because the exact integer register
contains the arithmetic transition; it is not a prior-free learned-circuit
result.

Protocol: values `0..95`, train depths `1..2`, held-out depths `3..4`, eight
base-16 digits, 2,000 steps, seeds 17/18. The body has `7,236,759` frozen
parameters; only `136,466` overlay parameters are trainable.

| Metric | Seed17 | Seed18 | Mean |
|---|---:|---:|---:|
| held-out overall | 58.8379% | 58.8867% | 58.8623% |
| depth-3 | 69.3359% | 69.1406% | 69.2383% |
| depth-4 | 48.3398% | 48.6328% | 48.4863% |

Fixed operation-wise multiply improved strongly over V0.304: depth-3
`50.3906%/43.7500%` (mean `47.0703%`) and depth-4
`18.3594%/22.6562%` (mean `20.5078%`) at 2k. The 5k seed17 extension reached
`100%` depth-3 and `56.25%` depth-4 ordinary multiply. Add/subtract were
unchanged because the body was frozen.

High-value multiply remained the bottleneck: at 2k it was
`27.7344%/27.9297%` at depth 3 but `0%/0%` at depth 4. At 5k seed17 it was
`100%` at depth 3 and still `0%` at depth 4. The overlay proves that exact
numeric readout helps, but it does not prove that the sparse circuits learned
the product transition.

## V0.307: prior-free learned 5k continuation

V0.304's algebraic Fourier state/query branch was trained from a fresh seed17
for 5,000 steps, with no exact integer output decoder. Held-out accuracy was
`41.1133%`, depth-3 `50.7813%`, depth-4 `31.4453%`, versus the matched V0.304
2k seed17 values `39.5020%/49.2188%/29.7852%`. This is only a small
`+1.61 pp` overall gain. Fixed multiply stayed near chance at
`1.1719%/1.3672%`, and high-value multiply stayed `0%/0%`.

Therefore longer training alone does not transfer the exact product state
into the learned recurrent output. It is not a capacity-scaling trigger.

## V0.308: operation-conditioned algebraic projection

V0.308 replaced the one shared algebraic Fourier projection with three
operation-specific projections for add, subtract, and multiply. The circuit
bank, router, active circuit count, and output codec were unchanged. This
tests whether a shared state/query bridge suppresses multiply information.

The seed17 2k pilot reached only `36.3281%` held-out accuracy,
`45.3125%` at depth 3, and `27.3438%` at depth 4, versus V0.304's
`39.5020%/49.2188%/29.7852%`. Fixed multiply was `1.7578%/1.1719%`, and
high-value multiply was `0%/0%`. The additional operation-conditioned maps
therefore do not solve the bottleneck and regress the positive V0.304 signal.

## Decision

- V0.306 is **retained only as a strong exact-numeric diagnostic**. It is not
  adopted as the Neural Engine quality result because its integer register
  supplies the known arithmetic transition.
- V0.307 confirms that more steps on the current learned state/query branch
  give only a small gain and leave multiply near chance.
- V0.308 is **rejected**; multiplying bridge projections is not the next path.
- The active learned problem is the operation-specific multiply transition
  and its depth-4 range-safe state, not generic router size. No 700M/1B scale
  follows. The next experiment should modify the learned write/dataflow
  contract, with exact-integer overlays kept as a separate upper-bound
  diagnostic.

## Artifacts

- V0.306 config:
  `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_operation_conditioned_product_algebraic_fourier_integer_overlay_multiply.yaml`
- V0.307 run:
  `results/runs/v0_307_algebraic_fourier_state_seed17_5000.json`
- V0.308 config:
  `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_operation_conditioned_product_operation_conditioned_algebraic_fourier_state.yaml`
- V0.306 run JSONs:
  `results/runs/v0_306_v304_integer_overlay_seed17_2000.json`,
  `results/runs/v0_306_v304_integer_overlay_seed18_2000.json`,
  `results/runs/v0_306_v304_integer_overlay_seed17_5000.json`
- V0.307/V0.308 operationwise JSONs are stored under
  `results/operationwise_*_v0_307_*` and `results/operationwise_*_v0_308_*`.
