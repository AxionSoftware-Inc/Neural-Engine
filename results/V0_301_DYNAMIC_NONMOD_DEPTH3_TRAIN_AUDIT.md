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

## Seed17 result

The run completed in `1,573.12 s` on CUDA. Final train loss was `6.32758`.

| Metric | V0.298 reference (train 1–2 / eval 3–4) | V0.301 (train 1–3 / eval 4) |
|---|---:|---:|
| held-out depth-4 overall accuracy | 14.3555% | 2.9297% |
| depth-4 add accuracy | 36.3281% | 8.5938% |
| depth-4 subtract accuracy | 34.9609% | 8.5938% |
| depth-4 multiply accuracy | 4.1016% | 4.6875% |

The operation-wise screen used 512 fixed homogeneous programs per operation.
The V0.301 digit accuracies for multiply were
`[65.04%, 33.79%, 15.04%, 12.50%, 10.35%, 13.48%, 25.98%, 78.52%]`.
Thus the multiply result is only slightly above the V0.298 reference and its
first internal digits remain poor. Add and subtract transfer regress sharply.

## Interpretation

The hypothesis is **rejected for adoption**. More training coverage at depth 3
did not unlock depth-4 composition. The result also does not support the idea
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

Seed18 replication was launched with the identical command and will be added
to this audit before finalizing the adoption decision.
