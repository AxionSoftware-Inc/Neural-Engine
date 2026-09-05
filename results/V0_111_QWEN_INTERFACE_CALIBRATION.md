# V0.111 Qwen Child Interface Calibration

## Question

Can a compact learned state/interface correction repair the two-layer child
handoff, after local distillation and joint end-to-end logit distillation
failed?

## Protocol

The V0.109 two-layer protocol is held fixed. Each 384-wide GELU child is
wrapped with a zero-initialized rank-64 residual interface:

`child(x) + Up(tanh(Down(child(x))))`

The two child blocks are locally distilled for 300 steps each, then jointly
refined for 300 steps against frozen Qwen teacher logits at temperature 2.0.
Only the two child blocks and their calibration parameters are trainable; all
Qwen attention and parent FFN parameters remain frozen.

```powershell
python -u benchmark_qwen_two_layer_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --first-layer 25 --second-layer 26 --inner-size 384 --child-kind gelu --calibration-rank 64 --batch-size 4 --sequence-length 64 --train-batches 8 --eval-batches 4 --steps 300 --learning-rate 0.003 --joint-steps 300 --joint-learning-rate 0.001 --joint-temperature 2.0 --seed 2026 --output results/runs/qwen_two_layer_calibration_rank64_seed2026.json
```

## Result

The alpha=0 two-child CE delta was **+0.2488**, with 92.58% teacher top-1
agreement. The uncalibrated GELU384 baseline was +0.2377. Each child used
920,960 scalar parameters including the rank-64 correction, versus 9,437,184
in the parent FFN.

The joint distillation loss did not descend monotonically (0.78 at step 1,
0.79 at 100, 0.89 at 200, 1.26 at 300), and final quality was slightly worse
than the no-calibration control.

## Decision

**This simple dense low-rank interface calibration is rejected.** It does not
repair the composition failure and the joint objective is not a stable remedy
for the current child representation. The remaining direction is sparse
functional specialization: route each token to a small subset of learned
child circuits instead of asking one compact block to approximate every local
Qwen FFN behavior.

## Artifact

- `benchmark_qwen_two_layer_transplant.py`
- `results/runs/qwen_two_layer_calibration_rank64_seed2026.json`
