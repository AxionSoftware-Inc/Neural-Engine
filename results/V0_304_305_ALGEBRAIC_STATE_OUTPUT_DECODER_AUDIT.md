# V0.304–V0.305 — algebraic Fourier state and terminal output decoder

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Question

V0.303's operation-conditioned product operator gave a reproducible gain on
ordinary held-out arithmetic, but multiplication and the `80..95` stress
screen remained near zero. V0.304 adds the fixed algebraic
`polynomial2_fourier` value/state features to the recurrent query while
keeping the learned terminal output path. V0.305 keeps V0.304 and replaces
the learned terminal state with a separate learned decoder fed directly by
the algebraic packet.

These are structured semantic sidecars, not evidence that the sparse circuit
bank itself learned exact arithmetic. They are evaluated as diagnostic/opt-in
branches and do not trigger 700M/1B scaling.

## Protocol

- V0.303 operation-conditioned operator-valued product body;
- 300M virtual factorized bank, 23,600 virtual circuits;
- values `0..95`, train depths `1..2`, held-out depths `3..4`;
- eight base-16 output digits, target offset `134,217,728`;
- batch `128`, 2,000 steps, examples-per-depth `1,024`, CUDA;
- matched fresh seeds `17` and `18` for V0.304;
- V0.305 output-decoder screen: seed `17`, same protocol.

## Parameter budget and end-to-end quality

| Variant | Seeds | Total params | Active estimate | Held-out acc. | Depth-3 | Depth-4 | CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| V0.304 algebraic state | 17/18 | 7,236,759 | 1,937,600 | 38.5986% | 48.2422% | 28.9551% | 8.0427 |
| V0.305 direct output decoder | 17 | 7,254,127 | 1,954,968 | 20.1660% | 24.2188% | 16.1133% | 8.2494 |

V0.304 seed-specific held-out accuracy was `39.5020%` (seed17) and
`37.6953%` (seed18), so the positive state/query signal repeats across seeds.
It is not a multiply solution: the gain is concentrated in add/subtract.

V0.305 regresses against its matched V0.304 seed17 run by `−19.3359 pp`
overall, `−25.0000 pp` at depth 3, and `−13.6719 pp` at depth 4. Directly
replacing the learned terminal representation is incompatible with the
current output codec/training objective.

## Fixed operation-wise screen

Each cell uses 512 deterministic homogeneous programs. V0.304 is the mean of
seeds 17/18; V0.305 is seed17.

| Operation | V0.304 d3 | V0.304 d4 | V0.305 d3 | V0.305 d4 |
|---|---:|---:|---:|---:|
| add | 87.3047% | 62.9883% | 37.3047% | 17.9688% |
| subtract | 99.5117% | 93.4570% | 23.6328% | 5.6641% |
| multiply | 1.5625% | 1.6602% | 3.9063% | 4.6875% |

The direct output decoder does not improve multiply and destroys the strong
add/subtract behavior supplied by the learned recurrent output path.

## High-value stress

The same evaluator was run on operands `80..95`:

| Operation | V0.304 d3 mean | V0.304 d4 mean | V0.305 d3 | V0.305 d4 |
|---|---:|---:|---:|---:|
| add | 26.3672% | 23.2422% | 0% | 0% |
| subtract | 50.0000% | 29.0039% | 0% | 0% |
| multiply | 0% | 0% | 0% | 0% |

The high-range multiply failure is unchanged. The remaining bottleneck is the
operation-specific product transition and range-safe output representation,
not simply router size or generic circuit capacity.

## Decision

- **V0.304 retained as a leading opt-in state/query branch.** Its two-seed
  2k improvement is real and reproducible, but comparison with V0.303 at 5k
  is not a matched training-budget claim.
- **V0.305 rejected for quality adoption.** Direct algebraic output decoding
  is not a safe replacement for the learned terminal output state.
- No default change and no 700M/1B scale-up follows. The next controlled
  direction is a multiply-specific range-safe codec/transition diagnostic,
  with exact-integer overlays reported separately from the prior-free
  learned circuit.

## Artifacts

- V0.304 config:
  `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_operation_conditioned_product_algebraic_fourier_state.yaml`
- V0.305 config:
  `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_operation_conditioned_product_algebraic_fourier_output_decoder.yaml`
- Runs:
  `results/runs/v0_304_algebraic_fourier_state_seed17_2000.json`,
  `results/runs/v0_304_algebraic_fourier_state_seed18_2000.json`,
  `results/runs/v0_305_algebraic_fourier_output_decoder_seed17_2000.json`
- Fixed-operation JSONs:
  `results/operationwise_fixed_checkpoint_eval_v0_304_seed17.json`,
  `results/operationwise_fixed_checkpoint_eval_v0_304_seed18.json`,
  `results/operationwise_fixed_checkpoint_eval_v0_305_seed17.json`
- High-value JSONs:
  `results/operationwise_high_value_checkpoint_eval_v0_304_seed17.json`,
  `results/operationwise_high_value_checkpoint_eval_v0_304_seed18.json`,
  `results/operationwise_high_value_checkpoint_eval_v0_305_seed17.json`
