# V0.273: carry-chain long-training control

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Purpose

V0.272's 2,000-step wide screen showed a small hard-accuracy gain for the
typed-digit carry chain, but the gain came with worse CE and was below the
adoption gate. V0.273 checks whether that result changes with the longer
training budget used by the stronger Native checkpoints. It compares a fresh
prior-free control and the carry-chain treatment at exactly 5,000 steps.

## Protocol

Both arms use depths 1--2 for training and 3--4 for evaluation, operands
`0..95`, batch size 128, the same 300M-virtual factorized circuit bank,
seeds 17/18, and 5,000 optimization steps. The treatment uses
`typed_digit_state=true`, four base-512 slots, a learned low-to-high carry
packet, and intermediate digit-contract loss weight `0.5`. The control has
none of those typed-state additions. The final comparison uses 1,024 identical
examples per held-out depth with evaluator seed 3102.

## Results

### Matched paired held-out evaluation

| variant | seed | all | d3 | d4 | eval loss |
|---|---:|---:|---:|---:|---:|
| control | 17 | 6.494% | 8.594% | 4.395% | 11.1817 |
| control | 18 | 7.422% | 9.668% | 5.176% | 11.2522 |
| carry-chain | 17 | 6.250% | 8.789% | 3.711% | 12.1195 |
| carry-chain | 18 | 6.104% | 8.105% | 4.102% | 11.3436 |
| **two-seed mean delta** |  | **−0.781 pp** | **−0.684 pp** | **−0.879 pp** | **+0.5146** |

The 5,000-step control mean (`6.958%`) is much higher than the 2,000-step
control mean (`2.148%`), confirming that training budget is a real confound
in the earlier screen. The carry-chain does not benefit similarly; it loses
to the matched control on all, depth-4, and CE, with both seeds agreeing on
the overall regression.

### Training and budget

Small final reports reached train accuracy `35.35%/41.60%` for the control and
`38.28%/40.43%` for the carry-chain. Thus the carry treatment can fit some
seen-depth examples, but that does not transfer to unseen depth. The
carry-chain has total `7,564,727` and estimated active `2,265,568` parameters;
the control has `7,483,463` and `2,184,304`. The treatment is slightly larger
in active compute while producing worse held-out quality.

Raw reports:

- `results/runs/v0_273_wide_control_seed17_5000.json`;
- `results/runs/v0_273_wide_control_seed18_5000.json`;
- `results/runs/v0_273_wide_carry_chain_seed17_5000.json`;
- `results/runs/v0_273_wide_carry_chain_seed18_5000.json`;
- `results/runs/v0_273_paired_control_seed17_large.json`;
- `results/runs/v0_273_paired_control_seed18_large.json`;
- `results/runs/v0_273_paired_carry_seed17_large.json`;
- `results/runs/v0_273_paired_carry_seed18_large.json`.

## Decision

**V0.273 rejects the carry-chain for quality adoption and scaling.** The
2,000-step positive screen was seed/budget noise; under a matched 5,000-step
budget the structured chain is consistently worse. This closes the current
typed-digit/carry-chain path as a solution to the capacity problem.

The implementation remains opt-in for reproducibility and future ablations.
The control's improvement with more training also means that any future
architecture claim must use a matched learning curve, not only a short screen.
Do not scale this path to 700M/1B. The next Native work should return to the
stronger algebraic/Fourier codec diagnostics or isolate operation-wise learned
state transition quality before another architecture change.

## Reproduction

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_prior_free.yaml --steps 5000 --seed 17 --run-id v0_273_wide_control_seed17_5000 --checkpoint results/checkpoints/v0_273_wide_control_seed17_5000.pt --heldout-depths --examples-per-depth 256 --train-value-min 0 --train-value-max 95 --eval-value-min 0 --eval-value-max 95
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_small_values0_7_typed_digit_carry_chain.yaml --steps 5000 --seed 17 --run-id v0_273_wide_carry_chain_seed17_5000 --checkpoint results/checkpoints/v0_273_wide_carry_chain_seed17_5000.pt --heldout-depths --examples-per-depth 256 --train-value-min 0 --train-value-max 95 --eval-value-min 0 --eval-value-max 95
python benchmark_prior_free_checkpoint_eval.py --checkpoint results/checkpoints/v0_273_wide_carry_chain_seed17_5000.pt --examples-per-depth 1024 --seed 3102 --value-min 0 --value-max 95 --split heldout --output results/runs/v0_273_paired_carry_seed17_large.json
```
