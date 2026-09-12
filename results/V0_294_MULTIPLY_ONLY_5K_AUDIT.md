# V0.294 — Matched 5k multiply-only training diagnostic

**Date:** 2026-09-12  
**Branch:** `exp/track-native-engine`  
**Status:** `REJECTED AS OPERATION-COVERAGE SOLUTION`

## Question

V0.292 identified multiplication as the dominant error source. V0.293's 2,000-
step multiply-only pilot did not improve it, but its shorter budget left a
possible confound. V0.294 repeats the same V0.290 numeric-bridge body for the
full 5,000-step budget, with every training program restricted to multiply.

## Results

| Metric | Result |
|---|---:|
| Train depth 1 accuracy | 96.8750% |
| Train depth 2 accuracy | 9.9609% |
| Random held-out depth 3 accuracy | 0.5859% |
| Random held-out depth 4 accuracy | 0.5859% |
| Overall random held-out accuracy | 0.5859% |
| Training time | 779.3 s |
| Estimated active parameters | 2,051,136 |

Fixed operation-wise evaluation on 512 examples per case:

| Operation | Depth 3 acc | Depth 4 acc | Depth 3 CE | Depth 4 CE |
|---|---:|---:|---:|---:|
| add | 0.0000% | 0.0000% | 36.9752 | 38.2500 |
| subtract | 0.0000% | 0.0000% | 90.6138 | 94.6182 |
| multiply | 3.9063% | 2.7344% | 31.4342 | 56.5179 |

The matched V0.290 numeric bridge reached `3.9063%`/`4.1992%` on the same
fixed multiply screen, so multiply-only training changes the result by
`0.0000 pp` at depth 3 and `−1.4648 pp` at depth 4. The 2,000-step V0.293 pilot
was `3.7109%`/`3.1250%`; the extra training does not produce a monotonic gain.
Train depth-1 fitting is strong, but train depth-2 remains only `9.9609%`,
showing that the learned transition cannot reliably compose even the first
multiply chain.

## Interpretation and decision

The operation-coverage hypothesis is rejected as the primary explanation for
the multiply ceiling. Giving the model only multiply examples and a matched
5,000-step budget still leaves held-out multiply near chance. The failure is in
the reusable multiply state transition/dataflow (and possibly its interaction
with the learned digit codec), not simply too few multiply samples.

The zero add/subtract scores are expected from this diagnostic's training
distribution and are not a claim that those operations are impossible.

**Decision:** no default change and no 700M/1B scaling. Close operation-only
sampling as the next fix. The next architecture test should make the typed
register's numeric state directly responsible for the final digit readout,
instead of using it only as an additive query feature; this isolates whether
the current learned accumulator is erasing the typed multiply state.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_multiply_only.yaml`
- `results/runs/v0_294_multiply_only_seed17_5000.json`
- `results/operationwise_fixed_checkpoint_eval_v0_294.json`
- `results/V0_292_OPERATIONWISE_FIXED_CHECKPOINT_AUDIT.md`

## Reproduction

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_multiply_only.yaml --steps 5000 --seed 17 --batch-size 128 --examples-per-depth 1024 --run-id v0_294_multiply_only_seed17_5000 --output results/runs --checkpoint results/checkpoints/v0_294_multiply_only_seed17_5000.pt --train-value-min 0 --train-value-max 95 --eval-value-min 0 --eval-value-max 95 --heldout-depths --device cuda --log-every 1000
python -u benchmark_operationwise_checkpoints.py --checkpoint results/checkpoints/v0_294_multiply_only_seed17_5000.pt --examples-per-case 512 --device cuda --output results/operationwise_fixed_checkpoint_eval_v0_294.json
```
