# V0.263–V0.264: prior dependence and prior-free learned-path audit

Date: 2026-09-11  
Branch: `exp/track-native-engine`

## Question

The V0.260/V0.261 checkpoints reached almost 100% with a full-range exact
integer output codec. V0.262 showed that turning the circuit residual on or
off made no measurable difference. This audit separates the exact numeric
codec, the existing learned readout, and a genuinely prior-free model trained
without the algebraic state path.

## V0.263 — inference ablation on the peak checkpoints

Script: `probe_prior_ablation.py`  
Raw report: `results/runs/v0_263_prior_ablation.json`  
Checkpoints: V0.260 seed17 and V0.261 seed18  
Evaluation: held-out depths 3–4, 256 examples per depth, operand ranges
`0..95` and fixed unseen `96`.

The three modes were:

1. `full`: configured polynomial/Fourier state plus exact integer codec;
2. `learned_readout`: same polynomial/Fourier state, exact integer codec
   disabled, learned factorized digit readout used;
3. `learned_no_prior`: exact codec disabled and algebraic state path disabled.

| checkpoint | range | full | learned readout | no prior |
|---|---:|---:|---:|---:|
| V0.260 seed17 | 0..95 | 99.609% | 82.227% | 0.000% |
| V0.261 seed18 | 0..95 | 100.000% | 79.102% | 0.000% |
| V0.260 seed17 | fixed 96 | 97.852% | 68.555% | 0.000% |
| V0.261 seed18 | fixed 96 | 97.852% | 62.305% | 0.000% |

Operation-wise on `0..95`, the learned readout was strong for addition and
subtraction (`100%` and `99.4–99.6%`) but multiply was only `11.328–11.523%`.
On fixed unseen `96`, learned multiply was `0%` in both seeds while the exact
codec was `100%` for each operation. Removing the algebraic state at inference
collapsed every operation to `0%` in this stress test.

This is not a fair retrained prior-free baseline: the V0.260/V0.261 learned
body was trained with the algebraic path available. It is nevertheless a
direct causal dependency test. The exact codec is responsible for most of the
peak quality, and the learned circuit path is not producing that quality.

## V0.264 — retrained prior-free control

Config: `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_prior_free.yaml`  
Checkpoints:

- `results/checkpoints/v0_264_prior_free_seed17_5000.pt`
- `results/checkpoints/v0_264_prior_free_seed18_5000.pt`

Both runs used the same 300M virtual circuit body, factorized four-digit
output, wide operand support `0..95`, training depths 1–2, held-out depths
3–4, 5,000 steps, batch size 128, and no algebraic state, exact integer
decoder, modular prior, or exact output packet.

| seed | train depth 1–2 | held-out all | depth 3 | depth 4 |
|---:|---:|---:|---:|---:|
| 17 | 35.352% | 6.836% | 10.547% | 3.125% |
| 18 | 41.602% | 6.445% | 7.422% | 4.297% |

The paired operation probe on the same checkpoints gave:

| seed | held-out add | held-out subtract | held-out multiply | fixed-96 all |
|---:|---:|---:|---:|---:|
| 17 | 14.453% | 12.695% | 5.273% | 11.914% |
| 18 | 13.867% | 9.570% | 6.250% | 16.406% |

The model has 7,483,463 stored parameters and an estimated 2,184,304 active
parameters per run, including 608,304 active circuit parameters. Training took
about 715–716 seconds per seed on CUDA. A `96..127` range was not used for
depth-4 because the generated products exceed the current classifier class
space; this is a protocol limitation, not a quality result.

## Decision

**V0.263:** retain as a diagnostic; do not claim the exact-codec score as a
learned sparse-circuit result.  
**V0.264:** reject as a quality candidate. Removing the prior and retraining
does not recover the arithmetic dataflow; the circuit/router path is currently
the bottleneck, not raw parameter count.

The next architecture test should make the recurrent state transition carry a
learned typed value contract, with operation-specific transition capacity and
an auxiliary state-target evaluation. Do not scale to 700M/1B yet, and do not
change the default model based on these opt-in experiments.

Reproduce:

```powershell
python probe_prior_ablation.py `
  --checkpoint results/checkpoints/v0_260_rank128_fullrange_codec_continued_seed17_15000.pt `
  --checkpoint results/checkpoints/v0_261_rank128_fullrange_codec_continued_seed18_15000.pt `
  --output results/runs/v0_263_prior_ablation.json `
  --examples-per-depth 256 --value-range 0 95 --value-range 96 96 --device auto

python train_dynamic_composition.py `
  --config configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_prior_free.yaml `
  --steps 5000 --seed 17 --run-id v0_264_prior_free_seed17_5000 `
  --output results/runs --checkpoint results/checkpoints/v0_264_prior_free_seed17_5000.pt `
  --examples-per-depth 256 --heldout-depths --train-value-min 0 `
  --train-value-max 95 --eval-value-min 0 --eval-value-max 95 --device auto
```
