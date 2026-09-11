# V0.267: bilinear transition all-depth control

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Purpose

V0.265 added a rank-16 operation-conditioned accumulator×operand residual at
the recurrent state-write boundary. Its depth-holdout screen showed a small
`+0.781 pp` mean gain, but did not establish that the transition itself was
useful. V0.267 trains the same variant on all depths 1–4, paired with the
prior-free V0.266 all-depth control, so depth extrapolation is not a confound.

## Protocol

Config: `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_prior_free_bilinear.yaml`

- same prior-free 300M-virtual body as V0.266;
- rank-16 operation-specific bilinear write transition;
- factorized four-digit output, base 512;
- non-modular operands `0..95`, target offset `2**27`;
- 5,000 steps, batch size 128, seeds 17 and 18;
- train and evaluation both use depths 1–4;
- no exact integer codec, algebraic state, modular prior, or exact packet.

The final reports evaluate 128 examples per depth. Because that evaluation is
small and showed seed variance, a second evaluation-only pass used the same
4,096 deterministic examples for all four checkpoints (1,024 per depth).
No weights were changed in that pass.

## Results

### Original run evaluation

| variant | seed | train all depths | eval all depths | d1 | d2 | d3 | d4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0.266 prior-free control | 17 | 12.012% | 11.719% | 26.172% | 11.328% | 6.250% | 3.125% |
| V0.266 prior-free control | 18 | 12.207% | 10.938% | 23.047% | 8.984% | 6.250% | 5.469% |
| V0.267 bilinear | 17 | 8.398% | 11.914% | 31.250% | 7.813% | 4.688% | 3.906% |
| V0.267 bilinear | 18 | 14.453% | 15.234% | 32.031% | 17.188% | 8.594% | 3.125% |

The small final batches give a nominal mean of `11.328%` for V0.266 and
`13.574%` for V0.267 (`+2.246 pp`), but the improvement is almost entirely
seed18; seed17 improves only `+0.195 pp`, and depth-4 is not improved.

### Larger paired evaluation

| variant | seed | all | d1 | d2 | d3 | d4 | eval loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0.266 prior-free control | 17 | 12.134% | 25.098% | 12.402% | 6.934% | 4.004% | 5.1032 |
| V0.266 prior-free control | 18 | 11.670% | 24.219% | 13.086% | 4.785% | 4.590% | 5.0567 |
| V0.267 bilinear | 17 | 9.912% | 21.777% | 9.277% | 4.590% | 4.004% | 5.4275 |
| V0.267 bilinear | 18 | 14.355% | 29.199% | 16.699% | 7.422% | 4.102% | 4.9836 |

On the larger identical batch the two-seed means are:

- V0.266: `11.902%`;
- V0.267: `12.134%`;
- difference: only `+0.232 pp`.

Thus the apparent `+2.246 pp` from the small evaluation is sampling/seed
variance, not a reliable quality gain. V0.267 also adds 56,448 stored
parameters (`7,483,463 → 7,539,911`) and substantially increases route
diversity: in the larger evaluation factor-row utilization rises from
`46.1%/42.9%` to `73.4%/83.1%` for seeds 17/18, while quality remains flat.
More routes are being used, but they do not form a useful learned arithmetic
state.

Raw reports:

- `results/runs/v0_266_prior_free_all_depths_seed17_5000.json`
- `results/runs/v0_266_prior_free_all_depths_seed18_5000.json`
- `results/runs/v0_267_bilinear_all_depths_seed17_5000.json`
- `results/runs/v0_267_bilinear_all_depths_seed18_5000.json`
- `results/runs/v0_267_paired_large_eval.json`

The evaluation helper is `benchmark_prior_free_checkpoint_eval.py`.

## Decision

**V0.267 is rejected for adoption and scaling.** The bilinear write residual
may alter routing and reduce loss for one seed, but it does not produce a
stable quality improvement, does not rescue deep composition, and does not
improve depth-4. Keep it as opt-in diagnostic code only. Do not increase its
rank and do not move to 700M/1B based on this result.

P-003/P-004 remain active. The next experiment should target a genuinely
typed, reusable value/carry contract at the recurrent boundary, with an
explicit stage/state diagnostic. It should be screened first on a small
support where the learned state can be inspected, then repeated on `0..95`.
