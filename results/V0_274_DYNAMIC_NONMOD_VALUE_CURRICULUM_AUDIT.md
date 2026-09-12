# V0.274 — Progressive non-modular value curriculum

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Question

The 300M virtual-bank, non-modular four-digit model was fitting seen depths
but transferred poorly to held-out depths on operands `0..95`. Earlier
changes to typed registers and carry transitions did not produce a stable
wide-range gain. This experiment asks whether the model needs the value range
to be opened progressively during training, without changing the Neural
Engine architecture or active-compute budget.

## Treatment

The treatment uses the existing prior-free factorized model and changes only
the training generator schedule:

```yaml
value_curriculum:
  - until_step: 1000
    value_min: 0
    value_max: 7
  - until_step: 2500
    value_min: 0
    value_max: 31
  - until_step: 5000
    value_min: 0
    value_max: 95
```

Training uses depths 1–2; evaluation uses held-out depths 3–4. Each result
uses 1,024 identical evaluation examples per held-out depth, seed `3102`, and
the same checkpoint evaluator. The model has `7,483,463` total parameters and
an estimated `2,184,304` active parameters; these numbers are unchanged from
the control.

The curriculum implementation is in `train_dynamic_composition.py`. It
validates ordered stages, recreates the generator at stage boundaries, and
stores the normalized schedule in the run JSON. The feature is opt-in; models
without `value_curriculum` keep the old training path.

## Paired results

| Seed | Full-range control accuracy | Curriculum accuracy | Delta (pp) | Control CE | Curriculum CE | Control d3/d4 | Curriculum d3/d4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 6.494% | 19.434% | +12.939 | 11.1817 | 9.9311 | 8.594% / 4.395% | 25.879% / 12.988% |
| 18 | 7.422% | 21.924% | +14.502 | 11.2522 | 8.8573 | 9.668% / 5.176% | 26.953% / 16.895% |
| 19 | 7.617% | 19.580% | +11.963 | 12.0508 | 9.8836 | 9.961% / 5.273% | 26.172% / 12.988% |
| 20 | 6.592% | 20.752% | +14.160 | 11.0429 | 9.7309 | 8.789% / 4.395% | 27.246% / 14.258% |
| **Mean** | **7.031%** | **20.422%** | **+13.391** | **11.3819** | **9.6007** | **9.253% / 4.810%** | **26.563% / 14.282%** |

Curriculum accuracy standard deviation across four seeds is `1.006 pp`,
versus `0.494 pp` for the full-range control. The quality gain is therefore
large and present in every seed, although the treatment has somewhat higher
seed spread.

## Route and capacity observations

Curriculum increased final-evaluation factor-row coverage in every recorded
seed: treatment used `72–113` distinct rows (`46.75–73.38%` of the 154-row
factor bank), while the corresponding controls used `35–66` rows
(`22.73–42.86%`). This is consistent with better training coverage, but it is
not itself a quality proof. The active path remains the same eight selected
circuits; no dense fallback or forced active-path increase was introduced.

## Interpretation

This is the first large positive signal in the wide `0..95` lane that survives
four seeds and a fixed 1,024-example paired evaluation. It localizes the
immediate failure more precisely: the learned value/state and composition
path can improve when the value geometry is introduced from easy to hard
ranges. The result is a training-protocol improvement, not yet evidence that
the current architecture will scale monotonically to 700M or 1B.

The curriculum does not reuse held-out-depth examples and does not add a
teacher, attention, modular arithmetic prior, extra parameters, or forced
active computation. It does change the training distribution, so an
out-of-distribution value-range screen is still required before calling it a
general arithmetic solution.

## Decision

**RETAINED AS THE LEADING OPT-IN TRAINING PROTOCOL.** It is not made a global
default and does not justify 700M/1B scaling by itself. The existing model
architecture and default configuration remain unchanged.

The next gate is causal schedule validation and range/depth generalization:

1. compare alternative boundaries (`0..15 → 0..47 → 0..95` and a smooth
   range schedule) at the same 5,000-step budget;
2. evaluate on a value range not used in the curriculum, not only held-out
   depth;
3. repeat the leading schedule on the next capacity only after it survives
   those screens;
4. keep the full-range and phase-reset controls in every comparison.

## Reproduction

Treatment seeds 17–20:

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_prior_free_value_curriculum.yaml --steps 5000 --seed 17 --run-id v0_274_wide_value_curriculum_seed17_5000 --checkpoint results/checkpoints/v0_274_wide_value_curriculum_seed17_5000.pt --heldout-depths --examples-per-depth 256 --train-value-min 0 --train-value-max 95 --eval-value-min 0 --eval-value-max 95 --log-every 500
```

Change `--seed` and run ID for seeds 18–20. Then use
`benchmark_prior_free_checkpoint_eval.py` with `--examples-per-depth 1024
--seed 3102 --value-min 0 --value-max 95 --split heldout`.

Raw paired JSONs:

- `results/runs/v0_274_paired_curriculum_seed17_large.json`
- `results/runs/v0_274_paired_curriculum_seed18_large.json`
- `results/runs/v0_275_paired_curriculum_seed19_large.json`
- `results/runs/v0_276_paired_curriculum_seed20_large.json`

Controls are the matched full-range runs recorded in V0.273/V0.275/V0.276.
The generator-reset-only control is documented separately in
`V0_277_DYNAMIC_NONMOD_VALUE_RESET_CONTROL_AUDIT.md`.
