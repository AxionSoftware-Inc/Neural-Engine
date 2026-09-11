# V0.271: explicit typed-digit recurrent state

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Purpose

V0.270 showed that a generic learned 16D numeric scratch state did not
survive the wide `0..95` task. V0.271 tests a more structured alternative:
the recurrent state is split into four base-512 digit slots, with separate
slot embeddings, an operation-conditioned transition, and an auxiliary
cross-entropy contract on the intermediate target digits. The sparse circuit
bank, factorized routing, and active circuit budget are otherwise unchanged.

This is a learned typed representation, not an exact arithmetic oracle. The
target offset is used only to align the state codec with the factorized output
layout; the circuit path remains active.

## Protocol

Config files:

- small screen: `configs/ne_dynamic_300m_nonmod_small_values0_7_typed_digit_state.yaml`;
- wide screen: `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_prior_free_typed_digit_state.yaml`.

Both use the prior-free 300M-virtual factorized model, depths 1--2 for
training and depths 3--4 for evaluation, 2,000 steps, batch size 128,
`typed_digit_dim=16`, four base-512 slots, and contract-loss weight `0.5`.
The matched controls are V0.268 for the small range and V0.270 for the wide
range. Final comparisons use 1,024 identical examples per held-out depth,
seed `3102`.

## Results

### Small `0..7` screen

| variant | seed | all | d3 | d4 | eval loss |
|---|---:|---:|---:|---:|---:|
| control | 17 | 65.186% | 77.246% | 53.125% | 6.2248 |
| control | 18 | 67.920% | 77.344% | 58.496% | 5.1184 |
| typed-digit state | 17 | 68.359% | 78.711% | 58.008% | 4.0204 |
| typed-digit state | 18 | 65.723% | 76.855% | 54.590% | 4.1984 |
| **two-seed mean delta** |  | **+0.488 pp** | **+0.488 pp** | **+0.488 pp** | **−1.5622** |

The lower CE is encouraging, but hard accuracy is below the `+2 pp`
adoption gate and the gain is not large relative to seed variance. The
The original 256-example run actually moved from a `67.773%` control mean to
a `66.113%` typed-state mean (`68.359%` and `63.867%` by seed), so this is
not treated as a reliable quality breakthrough.

### Wide `0..95` screen

| variant | seed | all | d3 | d4 | eval loss |
|---|---:|---:|---:|---:|---:|
| control | 17 | 2.148% | 2.734% | 1.563% | 11.1444 |
| control | 18 | 2.148% | 2.344% | 1.953% | 10.9715 |
| typed-digit state | 17 | 1.660% | 2.344% | 0.977% | 11.1816 |
| typed-digit state | 18 | 2.295% | 2.539% | 2.051% | 11.5314 |
| **two-seed mean delta** |  | **−0.171 pp** | **−0.098 pp** | **−0.244 pp** | **+0.2986** |

The training screen also exposed the failure: typed-digit train accuracy was
only `9.77%/5.47%` at the end of 2,000 steps, versus `10.0%/6.6%` for the
controls, and only roughly `17.5%--35.7%` of factor rows were used. This is a
learning/representation failure at the wide range, not evidence that a
larger circuit bank would help.

### Budget

The typed state adds `111,424` parameters: total `7,594,887` and estimated
active `2,295,728`, versus control total `7,483,463` and active `2,184,304`.
It does not change the number of selected circuits or the sparse execution
path, but it is not free: the extra state and contract heads are active on
every step.

Raw run reports:

- `results/runs/v0_271_small07_typed_digit_seed17_2000.json`;
- `results/runs/v0_271_small07_typed_digit_seed18_2000.json`;
- `results/runs/v0_271_paired_control_seed17_large.json`;
- `results/runs/v0_271_paired_control_seed18_large.json`;
- `results/runs/v0_271_paired_typed_seed17_large.json`;
- `results/runs/v0_271_paired_typed_seed18_large.json`;
- `results/runs/v0_271_wide_typed_digit_seed17_2000.json`;
- `results/runs/v0_271_wide_typed_digit_seed18_2000.json`;
- `results/runs/v0_271_wide_paired_control_seed17_large.json`;
- `results/runs/v0_271_wide_paired_control_seed18_large.json`;
- `results/runs/v0_271_wide_paired_typed_seed17_large.json`;
- `results/runs/v0_271_wide_paired_typed_seed18_large.json`.

## Decision

**V0.271 is rejected for default adoption and capacity scaling.** Explicit
typed-digit state gives a small, CE-positive signal on the narrow `0..7`
screen, but it does not transfer to `0..95` and slightly regresses the wide
paired mean. The current problem is deeper than adding a generic or typed
state packet: the learned circuit transition does not yet implement a
range-stable carry/composition operator.

Keep the implementation as opt-in diagnostic code for ablations. Do not
move to 700M/1B from this result. The leading quality reference remains the
algebraic/Fourier shared codec family; the next Native experiment should
either improve its cross-digit/carry interface or explicitly separate the
learned circuit transition from the numeric readout, with a matched control.

## Reproduction

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_small_values0_7_typed_digit_state.yaml --steps 2000 --seed 17 --run-id v0_271_small07_typed_digit_seed17_2000 --checkpoint results/checkpoints/v0_271_small07_typed_digit_seed17_2000.pt --heldout-depths --examples-per-depth 256 --train-value-min 0 --train-value-max 7 --eval-value-min 0 --eval-value-max 7
python benchmark_prior_free_checkpoint_eval.py --checkpoint results/checkpoints/v0_271_small07_typed_digit_seed17_2000.pt --examples-per-depth 1024 --seed 3102 --value-min 0 --value-max 7 --split heldout --output results/runs/v0_271_paired_typed_seed17_large.json
```
