# V0.287 — Scheduled teacher-forcing for the learned base-16 carry chain

**Date:** 2026-09-12  
**Status:** `REJECTED FOR QUALITY ADOPTION`; retained as a negative/diagnostic result  
**Branch:** `exp/track-native-engine`

## Question

V0.286 showed a small benefit from an eight-slot typed base-16 carry state, but
the learned transition remained weak. This experiment tests whether exposing the
ground-truth intermediate stage values early in training can teach that
transition, then be removed gradually so inference remains prior-free.

## Controlled setup

- Same 300M virtual factorized bank, rank 128, eight base-16 output heads and
  eight-slot typed carry state as V0.286.
- Same values `0..95`, train depths `1..2`, held-out depths `3..4`, batch `128`,
  `2,000` steps, `1,024` examples per depth, and seeds `17/18`.
- Teacher forcing is training-only: probability is linearly scheduled from
  `1.0` at step 1 to `0.0` at step 2,000. Evaluation passes no teacher targets
  and uses probability `0.0`.
- The teacher supplies only the current stage target values to the typed state;
  no teacher output or integer decoder is present at evaluation.
- The model body, routing policy, active parameter definition, and default
  inference path are otherwise unchanged.

## Results

| variant | seed | held-out acc | CE | depth 3 | depth 4 | train sec |
|---|---:|---:|---:|---:|---:|---:|
| V0.286 control | 17 | 3.1738% | 12.1439 | 3.7109% | 2.6367% | 297.1 |
| V0.286 control | 18 | 2.6855% | 12.4945 | 3.4180% | 1.9531% | 311.9 |
| V0.287 scheduled teacher | 17 | 3.5156% | 12.6233 | 4.3945% | 2.6367% | 378.9 |
| V0.287 scheduled teacher | 18 | 4.1016% | 11.8221 | 5.2734% | 2.9297% | 385.9 |

Two-seed means:

- Control → V0.287 accuracy: `2.9297% → 3.8086%` (`+0.8789 pp`).
- Control → V0.287 CE: `12.3192 → 12.2227` (`−0.0965`).
- Depth-3 accuracy: `3.5645% → 4.8340%` (`+1.2695 pp`).
- Depth-4 accuracy: `2.2949% → 2.7832%` (`+0.4883 pp`).
- Against the V0.286 typed-carry treatment, accuracy is only `+0.3174 pp`,
  while CE is `+0.2295` worse.
- Training time increases from `304.5 s` to `382.4 s` on average (`+25.6%`)
  versus the control. Active parameters remain `2,047,296`; teacher targets do
  not become an inference dependency.

## Interpretation

The schedule gives a real but small signal, mainly on depth 3 and seed 18. It
does not reliably convert supervision into a reusable learned transition:
seed 17 CE regresses, depth 4 moves only `+0.4883 pp`, and the improvement is
below the project’s `+2 pp` adoption gate. The extra training cost also matters.

**Decision:** do not make scheduled teacher forcing the default and do not scale
this variant to 700M/1B. Keep the implementation opt-in because it is useful as
a diagnostic and as a component for a stronger transition experiment. P-003 and
P-004 remain active. The next native test should target the transition itself
(for example an operation-conditioned reusable carry update or a compact
state-transition ablation), with teacher forcing used only as a diagnostic
control rather than as the proposed solution.

## Reproduction

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128.yaml --steps 2000 --seed 17 --batch-size 128 --examples-per-depth 1024 --run-id v0_287_base16_scheduled_teacher_seed17_2000 --output results/runs --checkpoint results/checkpoints/v0_287_base16_scheduled_teacher_seed17_2000.pt --train-value-min 0 --train-value-max 95 --eval-value-min 0 --eval-value-max 95 --heldout-depths --device cuda --log-every 500
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128.yaml --steps 2000 --seed 18 --batch-size 128 --examples-per-depth 1024 --run-id v0_287_base16_scheduled_teacher_seed18_2000 --output results/runs --checkpoint results/checkpoints/v0_287_base16_scheduled_teacher_seed18_2000.pt --train-value-min 0 --train-value-max 95 --eval-value-min 0 --eval-value-max 95 --heldout-depths --device cuda --log-every 500
```

Raw reports:

- `results/runs/v0_287_base16_scheduled_teacher_seed17_2000.json`
- `results/runs/v0_287_base16_scheduled_teacher_seed18_2000.json`

