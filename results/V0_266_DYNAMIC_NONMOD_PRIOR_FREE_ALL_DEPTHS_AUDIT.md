# V0.266: prior-free all-depth control

Date: 2026-09-11  
Branch: `exp/track-native-engine`

## Purpose

V0.264/V0.265 trained only depths 1–2 and evaluated depths 3–4. Their low
held-out scores could therefore have mixed two effects: a failed learned
state transition and untrained step embeddings/circuit behavior at deeper
steps. V0.266 removes that confound by training and evaluating the same
prior-free model on all depths 1–4.

## Protocol

Config: `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_prior_free.yaml`

- 300M virtual circuit body, 7,483,463 stored parameters;
- factorized four-digit output with base 512;
- operand range `0..95`, non-modular targets;
- no algebraic state, exact integer decoder, modular prior, or exact packet;
- 5,000 steps, batch size 128, two seeds;
- training and evaluation both use depths 1–4.

## Results

| seed | train all depths | eval all depths | eval d1 | eval d2 | eval d3 | eval d4 |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 12.012% | 11.719% | 26.172% | 11.328% | 6.250% | 3.125% |
| 18 | 12.207% | 10.938% | 23.047% | 8.984% | 6.250% | 5.469% |

The two-seed evaluation mean is `11.328%`. The paired operation probe on fresh
deterministic batches gave all-operation `5.664%/6.055%`, add
`12.500%/6.641%`, subtract `10.156%/8.398%`, and multiply `6.445%/6.445%`
for seed17/18. Fixed unseen-96 all-operation accuracy was `15.234%` and
`15.234%` in the same probe, but operation-specific fixed-96 results remained
zero; this is not evidence of extrapolation.

The runs took about 1,240 and 1,317 seconds on CUDA. Routing did not collapse
to one circuit: eval factor-row utilization was about `35%` and virtual-bank
utilization about `1%`, so the low score is not explained by a dead router
alone.

Raw reports:

- `results/runs/v0_266_prior_free_all_depths_seed17_5000.json`
- `results/runs/v0_266_prior_free_all_depths_seed18_5000.json`
- `results/runs/v0_266_prior_free_all_depths_operation_probe.json`

## Decision

**V0.266 is a negative control and is rejected as a quality candidate.**
Training on the evaluation depths does not recover the learned arithmetic
path: train and eval remain near `11–12%`, with deep multiply near chance.
This closes the explanation that V0.264/V0.265 failed only because depth-3/4
steps were unseen during training. The primary unresolved problem is the
learned value/state representation and its interface with sparse circuits,
not simply the router’s depth extrapolation or raw model capacity.

Next work should use a small, explicitly supervised learned value contract at
the recurrent state boundary, then test whether circuit residuals can operate
on that contract. Keep the exact codec branch as a separate hybrid baseline;
do not present its near-100% score as learned circuit quality and do not scale
to 700M/1B yet.

Reproduce:

```powershell
python train_dynamic_composition.py `
  --config configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_prior_free.yaml `
  --steps 5000 --seed 17 --run-id v0_266_prior_free_all_depths_seed17_5000 `
  --output results/runs --checkpoint results/checkpoints/v0_266_prior_free_all_depths_seed17_5000.pt `
  --examples-per-depth 256 --train-value-min 0 --train-value-max 95 `
  --eval-value-min 0 --eval-value-max 95 --device auto
```
