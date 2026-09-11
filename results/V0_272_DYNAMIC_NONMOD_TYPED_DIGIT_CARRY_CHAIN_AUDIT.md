# V0.272: typed-digit carry-chain transition

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Purpose

V0.271 showed that a flat learned transition over four typed digit slots did
not scale from `0..7` to `0..95`. V0.272 keeps the same typed state and
intermediate digit-contract loss, but processes slots from the
least-significant side to the most-significant side. A small learned carry
packet is passed between slots. The transition is still learned; no exact
addition or multiplication table is used.

## Protocol

Config: `configs/ne_dynamic_300m_nonmod_small_values0_7_typed_digit_carry_chain.yaml`.

The prior-free 300M-virtual factorized model uses depths 1--2 for training,
depths 3--4 for evaluation, 2,000 steps, batch size 128, two seeds (17/18),
and four base-512 digit slots with 16 dimensions per slot. The paired
evaluation uses 1,024 identical examples per held-out depth with seed 3102.
Controls are the V0.268 prior-free flat typed-state-free checkpoints.

## Results

### Small `0..7`

| variant | seed | all | d3 | d4 | eval loss |
|---|---:|---:|---:|---:|---:|
| control | 17 | 65.186% | 77.246% | 53.125% | 6.2248 |
| control | 18 | 67.920% | 77.344% | 58.496% | 5.1184 |
| carry-chain | 17 | 67.383% | 78.711% | 56.055% | 4.1990 |
| carry-chain | 18 | 66.943% | 77.637% | 56.250% | 4.5008 |
| **two-seed mean delta** |  | **+0.610 pp** | **+0.879 pp** | **+0.342 pp** | **−1.3217** |

Hard accuracy improves, but the gain is well below the `+2 pp` adoption gate.
The CE improvement is positive, so the chain is retained as a diagnostic
component rather than discarded as a completely useless representation.

### Wide `0..95`

| variant | seed | all | d3 | d4 | eval loss |
|---|---:|---:|---:|---:|---:|
| control | 17 | 2.148% | 2.734% | 1.563% | 11.1444 |
| control | 18 | 2.148% | 2.344% | 1.953% | 10.9715 |
| carry-chain | 17 | 2.686% | 3.516% | 1.855% | 11.8465 |
| carry-chain | 18 | 2.197% | 2.930% | 1.465% | 11.4107 |
| **two-seed mean delta** |  | **+0.293 pp** | **+0.684 pp** | **−0.098 pp** | **+0.5707** |

The chain gives a small hard-accuracy gain on wide support, but both CE and
depth-4 are mixed-to-worse. Training accuracy remains low (`10.35%/5.66%`),
and factor-row utilization stays sparse; the chain has not made the learned
transition range-stable.

### Budget

The carry-chain variant has total parameters `7,564,727` and estimated active
parameters `2,265,568`, slightly below the flat typed-state treatment because
the shared carry transition uses a smaller per-slot packet. It does not change
the selected circuit count. Its quality signal therefore comes from an
inductive-bias change, not a larger bank or active budget.

Raw reports:

- `results/runs/v0_272_small07_carry_chain_seed17_2000.json`;
- `results/runs/v0_272_small07_carry_chain_seed18_2000.json`;
- `results/runs/v0_272_paired_carry_seed17_large.json`;
- `results/runs/v0_272_paired_carry_seed18_large.json`;
- `results/runs/v0_272_wide_carry_chain_seed17_2000.json`;
- `results/runs/v0_272_wide_carry_chain_seed18_2000.json`;
- `results/runs/v0_272_wide_paired_carry_seed17_large.json`;
- `results/runs/v0_272_wide_paired_carry_seed18_large.json`.

## Decision

**V0.272 is retained as an opt-in diagnostic, but rejected for default
adoption and capacity scaling.** A directional carry-chain signal exists,
especially in hard accuracy, but it is small, CE-negative on the wide task,
and does not rescue the prior-free model. The result supports the idea that
state transition structure matters, while showing that a one-pass local
carry packet is insufficient for multiplication and depth transfer.

Do not move to 700M/1B from this result. The next useful experiment should
combine a structured transition with an explicit, learned cross-digit output
interface or test the transition in isolation with operation-wise diagnostics;
another generic state-width increase is not justified.

## Reproduction

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_small_values0_7_typed_digit_carry_chain.yaml --steps 2000 --seed 17 --run-id v0_272_small07_carry_chain_seed17_2000 --checkpoint results/checkpoints/v0_272_small07_carry_chain_seed17_2000.pt --heldout-depths --examples-per-depth 256 --train-value-min 0 --train-value-max 7 --eval-value-min 0 --eval-value-max 7
python benchmark_prior_free_checkpoint_eval.py --checkpoint results/checkpoints/v0_272_wide_carry_chain_seed17_2000.pt --examples-per-depth 1024 --seed 3102 --value-min 0 --value-max 95 --split heldout --output results/runs/v0_272_wide_paired_carry_seed17_large.json
```
