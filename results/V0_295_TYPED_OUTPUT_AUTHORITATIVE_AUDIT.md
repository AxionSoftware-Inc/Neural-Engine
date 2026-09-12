# V0.295 — Typed-register authoritative terminal readout

**Date:** 2026-09-12  
**Branch:** `exp/track-native-engine`  
**Status:** `REJECTED`

## Question

The V0.290 numeric bridge updates a typed digit state, but the normal forward
path uses that state only as an additive query feature. The learned accumulator
and its writer still produce the final logits. V0.295 makes the typed state's
learned digit heads the terminal output source, testing whether the learned
accumulator is erasing a useful typed state.

This is a readout-only architectural switch. The circuit bank, router, typed
carry transition, numeric multiply bridge, losses, and training distribution
are otherwise unchanged.

## Protocol and result

- Same 300M factorized body and active-8 route as V0.290;
- eight base-16 typed slots, numeric multiply bridge, contract loss `0.5`;
- non-modular values `0..95`, target offset `134217728`;
- train depths `1..2`, held-out depths `3..4`;
- mixed random operations, 2,000 steps, batch `128`, seed `17`, CUDA;
- direct terminal readout from typed digit heads, no exact arithmetic decoder.

| Metric | Result |
|---|---:|
| Train depth 1 accuracy | 2.9297% |
| Train depth 2 accuracy | 0.3906% |
| Random held-out depth 3 accuracy | 0.1953% |
| Random held-out depth 4 accuracy | 0.3906% |
| Overall random held-out accuracy | 0.2930% |
| Estimated active parameters | 2,051,136 |
| Training time | 357.1 s |

Fixed operation-wise evaluation (512 examples per case):

| Operation | Depth 3 acc | Depth 4 acc | Depth 3 CE | Depth 4 CE |
|---|---:|---:|---:|---:|
| add | 0.1953% | 0.1953% | 8.2631 | 12.4308 |
| subtract | 0.1953% | 0.1953% | 7.1726 | 7.9597 |
| multiply | 0.3906% | 0.0000% | 25.6863 | 54.4016 |

The authoritative readout collapses both fitting and held-out quality. It is
not a useful terminal codec in the current learned typed-state geometry. This
also shows that “typed state is being erased only at the final learned
accumulator” is not the full diagnosis: the typed state itself is not a stable
range-0..95 representation after learned recurrent updates.

**Decision:** reject as a default and quality path. Keep the flag as an opt-in
ablation. The next test should use the typed transition as a small,
operation-specific residual at the actual accumulator write boundary, rather
than bypassing the learned readout entirely.

## Artifacts

- `neural_engine/dynamic_register.py`
- `train_dynamic_composition.py`
- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_typed_output.yaml`
- `results/runs/v0_295_typed_output_authoritative_seed17_2000.json`
- `results/operationwise_fixed_checkpoint_eval_v0_295.json`
- `tests/test_dynamic_register.py`

## Reproduction

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_typed_output.yaml --steps 2000 --seed 17 --batch-size 128 --examples-per-depth 512 --run-id v0_295_typed_output_authoritative_seed17_2000 --output results/runs --checkpoint results/checkpoints/v0_295_typed_output_authoritative_seed17_2000.pt --train-value-min 0 --train-value-max 95 --eval-value-min 0 --eval-value-max 95 --heldout-depths --device cuda --log-every 500
python -u benchmark_operationwise_checkpoints.py --checkpoint results/checkpoints/v0_295_typed_output_authoritative_seed17_2000.pt --examples-per-case 512 --device cuda --output results/operationwise_fixed_checkpoint_eval_v0_295.json
```
