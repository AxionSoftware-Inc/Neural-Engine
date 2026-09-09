# V0.243 — CUDA atomic route-pack audit

**Date:** 2026-09-09  
**Status:** `PARITY PASS; STABLE SMALL OPT-IN SPEEDUP; DEFAULT UNCHANGED`  
**Branch:** `exp/track-runtime`

## Hypothesis

The grouped Qwen path previously built its expert-major workspace with a
Python/PyTorch sequence of `argsort`, `scatter_add`/`bincount`, cumulative
offset arithmetic, zero/empty workspace allocation, and `index_copy`. The
stage profile in V0.240 showed that this `pack` stage consumed about 46–49%
of the instrumented grouped path.

This probe replaces only that route packing stage with a small CUDA extension.
One CUDA block handles one selected `(token, slot)` pair, reserves an
expert-local row using an atomic counter, copies the hidden row to the
expert-major workspace, and returns the pair-to-workspace position. The
expert projections, correction, K-subset, accumulation, router, and model
weights are unchanged.

The implementation is intentionally opt-in. It currently supports contiguous
float32 CUDA tensors and uses a conservative `E * token_count` workspace
bound, which keeps the path CUDA-Graph safe but does not yet make it a final
production kernel.

## Protocol

Local `Qwen/Qwen3-0.6B`, layers `19–26`, `E=8`, `K=5`, rank-1 correction,
float32 CUDA, and the `600/600/200` trained-cascade recipe. Prefix length was
4; batch sizes were 1, 8, and 32. The baseline was
`grouped-adaptive`; the probe was `grouped-adaptive-atomic-pack`. Each path
used 20 CUDA-Graph warmups and 50 timing iterations. Two independent seeds
were measured with the same protocol.

Graph latency is in milliseconds. The ratio is
`atomic-pack / grouped-adaptive`.

| seed | batch | adaptive | atomic pack | ratio | change |
|---:|---:|---:|---:|---:|---:|
| 2026 | 1 | 11.162 | 10.880 | 0.9747 | −2.53% |
| 2026 | 8 | 16.972 | 16.648 | 0.9810 | −1.91% |
| 2026 | 32 | 34.588 | 34.096 | 0.9858 | −1.42% |
| 17 | 1 | 11.172 | 10.878 | 0.9737 | −2.63% |
| 17 | 8 | 16.957 | 16.700 | 0.9849 | −1.51% |
| 17 | 32 | 34.580 | 34.076 | 0.9854 | −1.46% |

Across the two seeds, the mean speedup is approximately 2.58% at B1, 1.71%
at B8, and 1.44% at B32. The effect is consistent in sign and remains after
the longer repeat, but it is still a small runtime improvement rather than a
major architectural result. The path remains slower than dense Qwen FFN at
these shapes; the probe improves the grouped implementation's overhead.

## Correctness and quality

Both seeds returned `PARITY_PASS`. CUDA-Graph/eager checks passed and the
maximum atomic-pack logit difference versus the existing single-token
reference stayed below `1.5e-5`, within the existing float32 grouped-path
rounding range. Eight-token greedy generation matched the grouped baseline
exactly for both seeds.

The trained-cascade quality numbers were unchanged in substance: CE deltas
were `+0.04109` (seed 2026) and `+0.03844` (seed 17). This is a runtime-only
change and makes no quality claim. The atomic counter makes expert-local row
order nondeterministic, so bitwise equality is not expected; the observed
numerical tolerance and exact generated tokens are the acceptance criteria.

## Decision

- Keep `grouped-adaptive-atomic-pack` as a validated opt-in runtime path.
- Do not replace the default yet: the gain is only about 1.4–2.6% and the
  grouped path is still slower than dense at the tested shapes.
- Do not interpret this as evidence that routing quality or model capacity
  improved; only the packing implementation changed.
- The next material runtime target is a fused route-aware kernel that avoids
  the intermediate expert-major workspace and combines packing, projection,
  correction, and token accumulation. The current extension is a useful
  baseline for measuring that larger step.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --warmup 20 --iterations 50 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-atomic-pack --experiment V0.243_grouped_atomic_pack_long --output results/runs/v0_243_grouped_atomic_pack_long_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --warmup 20 --iterations 50 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-atomic-pack --experiment V0.243_grouped_atomic_pack_long --output results/runs/v0_243_grouped_atomic_pack_long_seed17.json
```

## Artifacts

- `neural_engine/qwen_atomic_pack.cpp`
- `neural_engine/qwen_atomic_pack.cu`
- `neural_engine/qwen_atomic_pack.py`
- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `results/runs/v0_243_grouped_atomic_pack_long_seed2026.json`
- `results/runs/v0_243_grouped_atomic_pack_long_seed17.json`
