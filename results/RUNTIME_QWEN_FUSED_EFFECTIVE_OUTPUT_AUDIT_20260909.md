# V0.246 — Direct fused effective-output dispatch audit

**Date:** 2026-09-09  
**Status:** `SPEED HYPOTHESIS REJECTED; PARITY-SAFE OPT-IN RETAINED`  
**Branch:** `exp/track-runtime`

## Hypothesis

V0.245 showed that atomic packing and folded output correction compose to a
small 1.9–3.4% end-to-end improvement, but they still materialize the
expert-major grouped workspace. This probe uses the existing one-launch
selected-FFN CUDA kernel with the exact folded per-expert output weight. It
therefore bypasses grouped pack, grouped select/correction, and grouped
accumulation in one direct route-aware dispatch.

The probe is `fused-effective-output`. It changes no route IDs, active K,
router, model weights, or training recipe. It is an inference-only CUDA
float32 path.

## Integration correction

The first full-model attempt did not produce a result because a stale
Torch-extension cache lock from an earlier interrupted build blocked the
loader. The rerun used an isolated extension cache. The fused CUDA launch was
also bound to `getCurrentCUDAStream()` instead of the default stream so that
the path is compatible with the caller's CUDA Graph stream. A direct CUDA
smoke and the real Qwen parity run then completed successfully.

## Protocol

Local `Qwen/Qwen3-0.6B`, layers `19–26`, `E=8`, `K=5`, rank-1 correction,
float32 CUDA, and the `600/600/200` trained-cascade recipe. Prefix length was
4; batch sizes were 1, 8, and 32. Each path used 5 CUDA-Graph warmups and 10
timing iterations. Baseline: `grouped-adaptive`. Two seeds used the same
protocol.

Graph latency is in milliseconds. Ratio is `fused-effective-output /
grouped-adaptive`.

| seed | batch | adaptive | fused effective | ratio | change |
|---:|---:|---:|---:|---:|---:|
| 2026 | 1 | 11.189 | 15.103 | 1.350 | +35.0% |
| 2026 | 8 | 16.930 | 25.761 | 1.522 | +52.2% |
| 2026 | 32 | 34.534 | 83.001 | 2.403 | +140.3% |
| 17 | 1 | 11.289 | 15.189 | 1.345 | +34.5% |
| 17 | 8 | 16.986 | 25.834 | 1.521 | +52.1% |
| 17 | 32 | 34.516 | 82.639 | 2.394 | +139.4% |

The negative result is consistent across both seeds and all tested batches.
The direct kernel's one-block-per-selected-route implementation performs too
many serial hidden/group projection operations and does not exploit the
grouped GEMM reuse that the adaptive path retains.

## Correctness and quality

Both seeds returned `PARITY_PASS`. CUDA-Graph/eager checks passed, the maximum
logit differences stayed in the existing float32 tolerance range, and
eight-token greedy generation matched the grouped baseline exactly. The
quality recipe remained unchanged; CE deltas were `+0.04811` for seed 2026
and `+0.04090` for seed 17. This experiment makes no quality claim.

## Decision

- Reject `fused-effective-output` as a speed solution and do not make it the
  default.
- Retain the path as an opt-in parity probe and retain the current-stream fix
  for future CUDA Graph kernels.
- Do not pursue more variants of this naive one-block-per-route kernel.
- Keep V0.245's atomic-pack plus folded-output path as the best current runtime
  baseline. The next meaningful kernel must use tiled/grouped GEMM reuse or a
  true token-tile design; merely removing the workspace is insufficient.

## Reproduction

```text
$env:TORCH_EXTENSIONS_DIR = 'C:\\Users\\shaxz\\AppData\\Local\\Temp\\neural_engine_torch_ext_v246'
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --warmup 5 --iterations 10 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-atomic-effective-output --include-fused-effective-output --experiment V0.246_fused_effective_output_recovery --output results/runs/v0_246_fused_effective_output_recovery_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --warmup 5 --iterations 10 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-fused-effective-output --experiment V0.246_fused_effective_output_recovery --output results/runs/v0_246_fused_effective_output_recovery_seed17.json
```

## Artifacts

- `neural_engine/qwen_fused_dispatch.cu`
- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `results/runs/v0_246_fused_effective_output_recovery_seed2026.json`
- `results/runs/v0_246_fused_effective_output_recovery_seed17.json`
