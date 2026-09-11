# V0.265: learned operation-specific bilinear state transition

Date: 2026-09-11  
Branch: `exp/track-native-engine`

## Hypothesis

The existing product interaction was injected into the recurrent query, but
the state writer itself had no explicit learned interaction between the
current accumulator and the next operand. V0.265 adds a small
operation-conditioned low-rank bilinear residual at the write boundary:

```text
z = (accumulator @ A_operation) * (operand @ B_operation)
write_input += scale * GELU(z @ U_operation + bias_operation)
```

This is a learned transition, not the exact integer/algebraic prior. The
default remains unchanged; the new branch is opt-in.

## Implementation

Files:

- `neural_engine/dynamic_register.py`
- `train_dynamic_composition.py`
- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_prior_free_bilinear.yaml`
- `tests/test_dynamic_register.py`

The rank-16 transition adds `56,448` stored parameters to the prior-free
model (`7,539,911` total). The two-seed runs used the same wide-support,
depth-holdout protocol as V0.264: operand range `0..95`, train depths 1–2,
held-out depths 3–4, 5,000 steps, batch size 128, and 256 examples per depth.

## Results

| variant | seed | train depth 1–2 | held-out all | depth 3 | depth 4 |
|---|---:|---:|---:|---:|---:|
| V0.264 prior-free | 17 | 35.352% | 6.836% | 9.766% | 3.906% |
| V0.265 bilinear | 17 | 38.281% | 8.203% | 10.156% | 6.250% |
| V0.264 prior-free | 18 | 41.602% | 6.445% | 7.813% | 5.078% |
| V0.265 bilinear | 18 | 43.750% | 6.641% | 7.813% | 5.469% |

Two-seed held-out mean is `7.422%` for V0.265 versus `6.641%` for V0.264,
an improvement of `+0.781 pp`, below the `+2 pp` adoption gate. The V0.265
train/eval runs took about 723–728 seconds per seed on CUDA.

The separate paired operation probe (same script and evaluation protocol,
different deterministic probe batches) reported:

| seed | all | add | subtract | multiply | fixed-96 all |
|---:|---:|---:|---:|---:|---:|
| 17 | 4.883% | 18.359% | 12.500% | 5.664% | 13.086% |
| 18 | 7.227% | 15.625% | 14.453% | 6.445% | 11.133% |

Raw report: `results/runs/v0_265_bilinear_operation_probe.json`. The probe
confirms that unseen fixed-96 multiplication remains `0%` in both seeds.
The small absolute differences from the training report are expected because
the operation probe uses fresh deterministic seeds; the main matched
comparison is the train/eval report pair:

- `results/runs/v0_265_bilinear_seed17_5000.json`
- `results/runs/v0_265_bilinear_seed18_5000.json`

## Decision

**RETAIN AS OPT-IN DIAGNOSTIC, NOT ADOPTED.** The write-boundary bilinear
interaction gives a reproducible but small quality signal and is more
architecturally relevant than raw capacity scaling. It does not solve the
state contract: deep multiply remains near chance and unseen values remain
unusable. Do not increase bilinear rank or scale yet.

The next control should train prior-free variants on all depths (not only
depth-holdout) to separate untrained step extrapolation from genuine learned
transition failure. Only after that control should a larger typed-state
transition be considered. No 700M/1B scaling follows from V0.265.

Reproduce:

```powershell
python train_dynamic_composition.py `
  --config configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_prior_free_bilinear.yaml `
  --steps 5000 --seed 17 --run-id v0_265_bilinear_seed17_5000 `
  --output results/runs --checkpoint results/checkpoints/v0_265_bilinear_seed17_5000.pt `
  --examples-per-depth 256 --heldout-depths --train-value-min 0 `
  --train-value-max 95 --eval-value-min 0 --eval-value-max 95 --device auto
```
