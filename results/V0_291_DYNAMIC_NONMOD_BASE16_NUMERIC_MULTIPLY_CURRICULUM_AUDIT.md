# V0.291 — Numeric multiply bridge × progressive value curriculum

**Date:** 2026-09-12  
**Branch:** `exp/track-native-engine`  
**Status:** `REJECTED FOR QUALITY ADOPTION; RETAINED AS OPT-IN DIAGNOSTIC`

## Question

V0.290's learned numeric cross-digit multiply bridge was the strongest typed
state candidate, but it still trailed the plain base-16 control. V0.291 tested
whether the progressive value curriculum that helped the earlier typed-carry
variant would make the numeric bridge easier to learn.

The curriculum exposed values in three stages: `0..7` through step 1,000,
`0..31` through step 2,500, and the full `0..95` range through step 5,000.
The numeric bridge, typed carry, model body, router, loss weights, and inference
path were otherwise unchanged. No teacher forcing or exact arithmetic decoder
was used.

## Protocol

- Same 300M virtual-bank factorized model and active-8 route as V0.290;
- eight base-16 output digits, typed 8×16D carry state, contract loss `0.5`;
- learned soft digit values feeding the numeric partial-product convolution;
- progressive value curriculum `0..7 → 0..31 → 0..95`;
- train depths `1..2`, held-out depths `3..4`, values `0..95`;
- 5,000 fresh steps, batch `128`, 1,024 examples per held-out depth;
- CUDA, seeds `17/18`, compact factorized evaluation.

## Results

| Variant | Seed | Held-out acc | CE | Depth 3 | Depth 4 | Train sec |
|---|---:|---:|---:|---:|---:|---:|
| Numeric bridge + curriculum | 17 | 18.7012% | 10.7313 | 24.5117% | 12.8906% | 1119.2 |
| Numeric bridge + curriculum | 18 | 17.8711% | 12.8574 | 24.5117% | 11.2305% | 1059.2 |
| **Two-seed mean** | — | **18.2861%** | **11.7944** | **24.5117%** | **12.0605%** | **1089.2** |

Matched references:

| Variant | Held-out acc | CE | Depth 3 | Depth 4 |
|---|---:|---:|---:|---:|
| Base-16 control, plain (V0.288) | 20.7764% | 12.6073 | 26.4160% | 15.1367% |
| Typed carry, plain (V0.288) | 18.0176% | 12.0666 | 23.7305% | 12.3047% |
| Numeric bridge, no curriculum (V0.290) | 20.5078% | 11.7713 | 26.6602% | 14.3555% |
| Numeric bridge + curriculum (V0.291) | 18.2861% | 11.7944 | 24.5117% | 12.0605% |

Relative to V0.290, the curriculum changes held-out accuracy by `−2.2217 pp`,
CE by `+0.0231`, depth-3 accuracy by `−2.1484 pp`, and depth-4 accuracy by
`−2.2949 pp`. Relative to the plain control, the gap becomes `−2.4902 pp` in
accuracy and `−3.0762 pp` at depth 4. The near-flat mean CE does not translate into
better hard selection; this is a calibration/decision-quality regression, not
a quality improvement.

The estimated active parameter count stays at `2,051,136`. Mean training time
is `1089.2 s`, about `1.7%` above V0.290 and about `45.0%` above the plain
control screen. The experiment therefore adds cost without recovering quality.

## Interpretation and decision

The curriculum is not a generally helpful wrapper around the numeric bridge.
It helped the earlier typed-carry variant relative to its own weak baseline,
but that effect did not transfer to the numeric bridge. A plausible explanation
is that the staged value distribution changes the learned soft-digit geometry
and leaves too little full-range training budget for the bridge; this is a
hypothesis, not a proven causal diagnosis.

**Decision:** reject V0.291 for default adoption and scaling. Keep the config
and raw runs as an opt-in negative control. Retain V0.290 as the leading typed
state candidate, but do not scale to 700M/1B on this evidence. The next test
should be operation-wise evaluation of the existing checkpoints, especially
multiply versus add/subtract, before introducing another architecture change.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_curriculum.yaml`
- raw reports under `results/runs/v0_291_base16_numeric_multiply_convolution_curriculum_*_5000.json`
- `results/V0_290_DYNAMIC_NONMOD_BASE16_NUMERIC_MULTIPLY_CONVOLUTION_AUDIT.md`

## Reproduction

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_curriculum.yaml --steps 5000 --seed 17 --batch-size 128 --examples-per-depth 1024 --run-id v0_291_base16_numeric_multiply_convolution_curriculum_seed17_5000 --output results/runs --checkpoint results/checkpoints/v0_291_base16_numeric_multiply_convolution_curriculum_seed17_5000.pt --train-value-min 0 --train-value-max 95 --eval-value-min 0 --eval-value-max 95 --heldout-depths --device cuda --log-every 1000
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_curriculum.yaml --steps 5000 --seed 18 --batch-size 128 --examples-per-depth 1024 --run-id v0_291_base16_numeric_multiply_convolution_curriculum_seed18_5000 --output results/runs --checkpoint results/checkpoints/v0_291_base16_numeric_multiply_convolution_curriculum_seed18_5000.pt --train-value-min 0 --train-value-max 95 --eval-value-min 0 --eval-value-max 95 --heldout-depths --device cuda --log-every 1000
```
