# V0.301 — Depth-3 Training Coverage Audit

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Question

V0.298 was trained on depths 1–2 and evaluated on depths 3–4. Since the
dominant failure was depth-4 composition, this pilot trained on depths 1–3
and evaluated only on unseen depth 4. The hypothesis was that giving the
recurrent state one additional training depth would improve depth-4 transfer,
especially for multiplication.

This is a depth-coverage experiment, not a new router or a capacity increase.
The frozen architecture is the V0.298 numeric-convolution typed-carry model:

- 300M-class factorized circuit bank;
- base-16 typed digit state with numeric multiply convolution;
- train values `0..95`, eval values `0..95`;
- train depths `1..3`, held-out depth `4`;
- 5,000 steps, batch size 128, 1,024 examples per depth;
- seed17 first, seed18 matched replication;
- no curriculum, no teacher forcing, no output authority, no write residual.

## Results

Both runs completed in about 26 minutes on CUDA. Final train losses were
`6.32758` (seed17) and `6.05121` (seed18).

| Metric | V0.298 reference | V0.301 seed17 | V0.301 seed18 | V0.301 mean |
|---|---:|---:|---:|---:|
| held-out depth-4 overall accuracy | 14.3555% | 2.9297% | 3.0273% | 2.9785% |
| held-out depth-4 CE | 11.7713 | 7.4911 | 7.5344 | 7.5128 |
| depth-4 add accuracy | 36.6211% | 8.5938% | 10.5469% | 9.5703% |
| depth-4 subtract accuracy | 30.8594% | 8.5938% | 9.3750% | 8.9844% |
| depth-4 multiply accuracy | 4.1992% | 4.6875% | 7.4219% | 6.0547% |

The matched two-seed deltas versus V0.298 are `−11.3770 pp` overall,
`−27.0508 pp` add, `−21.8750 pp` subtract, and `+1.8555 pp` multiply in the
fixed operation-wise screen. The small multiply increase is not enough to
offset the broad regression and does not cross the quality gate.

The operation-wise screen used 512 fixed homogeneous programs per operation
and seed. The V0.301 seed17 digit accuracies for multiply were
`[65.04%, 33.79%, 15.04%, 12.50%, 10.35%, 13.48%, 25.98%, 78.52%]`; seed18
was `[64.84%, 31.45%, 16.80%, 12.50%, 14.26%, 14.26%, 24.02%, 80.47%]`.
The first internal digits remain poor. Add and subtract transfer regress
sharply in both seeds.

## Interpretation

The hypothesis is **rejected for adoption**. More training coverage at depth 3
did not unlock depth-4 composition in either seed. The result also does not support the idea
that the earlier failure was simply “not enough steps on depth 3”. In this
protocol, the extra depth changes the training distribution and harms the
previously stronger depth-4 add/subtract transfer while leaving multiply near
chance.

The current evidence points to a deeper state-transition/value-codec problem:

1. multiply remains the dominant bottleneck even when depth 3 is trained;
2. the numeric multiply bridge helps the typed baseline at shallow/ordinary
   screens but does not compose on depth-4 fixed multiplication;
3. output authority, write residuals, multiply-only training, and now depth-3
   coverage have all failed to produce a reliable multiply gain;
4. therefore another router-only change or a larger bank is not justified.

## Decision

- Keep V0.298 as the current learned-path reference.
- Keep V0.301 as a rejected, reproducible depth-coverage experiment.
- Do not scale this protocol to 700M/1B.
- Next work should target an explicit operation-conditioned algebraic/value
  transition that preserves the typed carry/product structure before the final
  factorized readout. It must be compared against V0.298 on fixed add,
  subtract, and multiply depth-4 screens, including high-value operands.

## Artifacts

- Training config:
  `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_train_depth3.yaml`
- Seed17 run:
  `results/runs/v0_301_train_depth3_seed17_5000.json`
- Seed17 checkpoint:
  `results/checkpoints/v0_301_train_depth3_seed17_5000.pt`
- Seed17 operation-wise evaluation:
  `results/operationwise_fixed_checkpoint_eval_v0_301.json`
- Seed18 run:
  `results/runs/v0_301_train_depth3_seed18_5000.json`
- Seed18 checkpoint:
  `results/checkpoints/v0_301_train_depth3_seed18_5000.pt`
- Seed18 operation-wise evaluation:
  `results/operationwise_fixed_checkpoint_eval_v0_301_seed18.json`

The operation-wise evaluator now de-duplicates the depth list when
`train_max_ops + 1 == max_ops`, preventing duplicate records in this protocol.
