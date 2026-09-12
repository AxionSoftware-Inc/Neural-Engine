# V0.293 — Multiply-only training distribution pilot

**Date:** 2026-09-12  
**Branch:** `exp/track-native-engine`  
**Status:** `PILOT; NOT YET A FINAL REJECTION`

## Question

V0.292 showed that held-out multiply accuracy is near chance while add and
subtract are materially stronger. This pilot tests whether multiply is failing
only because random mixed-operation training supplies too little multiply
coverage. The model body is V0.290's learned numeric multiply bridge; only the
training generator is changed to emit multiply operations for every executed
stage.

## Protocol

- Same 300M factorized model, typed carry state, numeric multiply bridge, and
  active-8 sparse route as V0.290;
- non-modular values `0..95`, target offset `134217728`;
- training depths `1..2`, held-out depths `3..4`;
- every training operation is `multiply`;
- 2,000 steps, batch `128`, seed `17`, CUDA;
- evaluation uses the standard random-operation held-out report plus the fixed
  operation-wise benchmark with 512 examples per case.

## Results

Training report:

| Metric | Result |
|---|---:|
| Train depth 1 accuracy | 60.1563% |
| Train depth 2 accuracy | 7.8125% |
| Random held-out depth 3 accuracy | 0.1953% |
| Random held-out depth 4 accuracy | 0.7813% |
| Overall random held-out accuracy | 0.4883% |
| Mean training time | 307.2 s |

Fixed operation-wise evaluation:

| Operation | Depth 3 acc | Depth 4 acc | Depth 3 CE | Depth 4 CE |
|---|---:|---:|---:|---:|
| add | 0.0000% | 0.0000% | 26.9404 | 27.8327 |
| subtract | 0.0000% | 0.0000% | 74.2686 | 75.7657 |
| multiply | 3.7109% | 3.1250% | 26.5183 | 48.6491 |

The fixed multiply result remains near the earlier V0.288/V0.290 range
(`3.6133%/3.6133%` and `3.9063%/4.1992%` at depths 3/4). It does not show an
early benefit from giving multiply all training examples. Add/subtract collapse
as expected because they receive no training signal; those zeros are not a
quality claim about the architecture.

## Interpretation and decision

This pilot weakens the simple operation-coverage explanation: concentrating all
training on multiply did not immediately produce a transferable multiply
circuit. It is not a definitive final rejection because the pilot used 2,000
steps rather than V0.290's 5,000-step budget. A matched full-budget multiply-
only run is required before closing the hypothesis.

**Decision:** retain as a diagnostic only; no default change and no scaling.
Run the same configuration for 5,000 steps at seed 17. If the fixed multiply
accuracy remains near chance, close operation coverage as the primary cause and
return to a genuinely different multiply state transition/dataflow.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_multiply_only.yaml`
- `data/dynamic_composition.py` (`fixed_operation` support)
- `train_dynamic_composition.py` (`train_fixed_operation` support)
- `results/runs/v0_293_multiply_only_seed17_2000.json`
- `results/operationwise_fixed_checkpoint_eval_v0_293.json`
- `tests/test_operationwise_checkpoint_eval.py`

## Reproduction

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_eight_digit_base16_typed_carry_rank128_multiply_numeric_convolution_multiply_only.yaml --steps 2000 --seed 17 --batch-size 128 --examples-per-depth 512 --run-id v0_293_multiply_only_seed17_2000 --output results/runs --checkpoint results/checkpoints/v0_293_multiply_only_seed17_2000.pt --train-value-min 0 --train-value-max 95 --eval-value-min 0 --eval-value-max 95 --heldout-depths --device cuda --log-every 500
python -u benchmark_operationwise_checkpoints.py --checkpoint results/checkpoints/v0_293_multiply_only_seed17_2000.pt --examples-per-case 512 --device cuda --output results/operationwise_fixed_checkpoint_eval_v0_293.json
```
