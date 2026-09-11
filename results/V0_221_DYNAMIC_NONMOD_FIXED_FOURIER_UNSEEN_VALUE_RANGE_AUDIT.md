# V0.221 — fixed Fourier value encoder on unseen operand range

**Date:** 2026-09-11  
**Status:** `REJECTED AS DIRECT OOD FIX`

## Question

V0.220 showed that the four-digit base-512 output codec fits operands `0..31`
but transfers poorly to unseen operands `32..63`. This experiment tests whether
the learned input value encoder is the main cause. The existing
`fixed_fourier` encoder was enabled while keeping the model body, circuit bank,
router, output codec, target offset, optimizer, and evaluation protocol fixed.

## Protocol

- four base-512 output digits, shared output rank `128`;
- `value_encoder_mode: fixed_fourier`;
- training operands `0..31`, held-out operands `32..63`;
- training depths `1..2`, held-out depths `3..4`;
- safe target offset `33,554,432`;
- 3000 steps, batch `128`, seeds `17` and `18`;
- factorized bank with `23,600` virtual circuits and active-8 routing;
- no new active circuit computation: the fixed Fourier encoder is an existing
  input representation mode, not an additional circuit path.

The safe offset is required because unmodulated multiply/subtract chains can
produce negative raw targets. The earlier V0.220 unsafe-offset attempt was
stopped by the range guard and is not part of this comparison.

## Results

| Seed | Train accuracy | Unseen-range accuracy | Depth 3 | Depth 4 | Unseen-range CE | Total params | Active estimate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 98.63% | 44.73% | 48.83% | 40.62% | 10.7553 | 7,464,591 | 2,165,432 |
| 18 | 100.00% | 47.85% | 52.34% | 43.36% | 10.6107 | 7,464,591 | 2,165,432 |
| **Mean** | **99.32%** | **46.29%** | **50.59%** | **41.99%** | **10.6830** | **7,464,591** | **2,165,432** |

V0.220's matched learned-encoder control reached `51.66%` mean unseen-range
accuracy, `55.47%` depth-3, `47.85%` depth-4, and `10.1647` mean CE. Therefore
fixed Fourier changes the result by:

- overall accuracy: `−5.37 pp`;
- depth-3 accuracy: `−4.88 pp`;
- depth-4 accuracy: `−5.86 pp`;
- CE: `+0.5183` (worse).

The fixed encoder still fits the seen range, so this is not a basic optimizer
failure. It also does not recover the missing value-range transfer; it is
strictly worse than the learned-encoder control on this gate.

## Decision

`fixed_fourier` is **rejected as the direct solution** to P-003's unseen-value
generalization problem. It remains available as an opt-in diagnostic because
it can be useful for separating learned embedding effects from state/readout
effects, but it must not replace the leading full-range-trained four-digit
candidate and must not become the default.

The evidence shifts the focus away from the input embedding alone. The open
problem is more likely the value/carry information preserved through recurrent
state, circuit composition, and the final digit readout. Increasing capacity
to 700M/1B is not justified by this result.

## Reproduction

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_31_eval32_63_four_digit_base512_rank128_fixedfourier_safeoffset.yaml --steps 3000 --seed 17 --run-id nonmod_train0_31_eval32_63_four_digit_base512_rank128_fixedfourier_safeoffset_seed17_3000 --checkpoint results/checkpoints/nonmod_train0_31_eval32_63_four_digit_base512_rank128_fixedfourier_safeoffset_seed17_3000.pt --output results/runs --examples-per-depth 256 --log-every 500 --heldout-depths --train-value-min 0 --train-value-max 31 --eval-value-min 32 --eval-value-max 63
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_31_eval32_63_four_digit_base512_rank128_fixedfourier_safeoffset.yaml --steps 3000 --seed 18 --run-id nonmod_train0_31_eval32_63_four_digit_base512_rank128_fixedfourier_safeoffset_seed18_3000 --checkpoint results/checkpoints/nonmod_train0_31_eval32_63_four_digit_base512_rank128_fixedfourier_safeoffset_seed18_3000.pt --output results/runs --examples-per-depth 256 --log-every 500 --heldout-depths --train-value-min 0 --train-value-max 31 --eval-value-min 32 --eval-value-max 63
```

## Raw runs

- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_fixedfourier_safeoffset_seed17_3000.json`
- `results/runs/nonmod_train0_31_eval32_63_four_digit_base512_rank128_fixedfourier_safeoffset_seed18_3000.json`
