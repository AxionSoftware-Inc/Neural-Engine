# V0.241 — Uninitialized-padding grouped dispatch probe

**Date:** 2026-09-09  
**Status:** `REJECTED AS DEFAULT; MICRO-OPTIMIZATION ONLY`  
**Branch:** `exp/track-runtime`

## Hypothesis

V0.240 showed that expert-major `pack` is the largest grouped-dispatch
stage. The current implementation zero-fills the full padded
`[num_experts, max_count, hidden]` workspace before writing the selected rows.
For a selected row, only its own indexed output is consumed later; padded rows
are never selected. The probe therefore uses `torch.empty` for this workspace
and writes the same selected rows, avoiding the padding memset.

This changes only workspace initialization. It does not alter routing, active
K, expert weights, correction, accumulation, or the returned selected rows.

## Protocol

Local `Qwen/Qwen3-0.6B`, layers `19–26`, `E=8`, `K=5`, rank-1 correction,
float32 CUDA, and the same `600/600/200` trained-cascade recipe as V0.238–
V0.240. Prefix length was 4; batch sizes were 1, 8, and 32. Each path used
10 CUDA-Graph warmups and 20 timing iterations. The baseline was the current
`grouped-adaptive` policy; the probe was identical except for the
uninitialized grouped workspace.

## Results

Graph latency is in milliseconds. The ratio is `nozero / adaptive`.

| seed | batch | adaptive | nozero | ratio | change |
|---:|---:|---:|---:|---:|---:|
| 2026 | 1 | 11.179 | 11.165 | 0.9988 | −0.12% |
| 2026 | 8 | 17.030 | 17.006 | 0.9986 | −0.14% |
| 2026 | 32 | 34.607 | 34.415 | 0.9944 | −0.56% |
| 17 | 1 | 11.256 | 11.300 | 1.0040 | +0.40% |
| 17 | 8 | 16.929 | 16.948 | 1.0011 | +0.11% |
| 17 | 32 | 34.466 | 34.415 | 0.9985 | −0.15% |

Both seeds passed CUDA Graph/eager parity and eight-token greedy generation
matched the adaptive baseline exactly. Quality remained the same trained
recipe: CE delta `+0.04531` (seed 2026) and `+0.03977` (seed 17). The runtime
change is inconsistent and below 1%; it is within normal timing noise for
this end-to-end path.

## Decision

- Do not promote `grouped-adaptive-nozero` to the default.
- Keep the implementation as an opt-in probe because it is numerically safe
  and may be useful on hardware where workspace memset is expensive.
- The V0.240 conclusion stands: the material target is a fused,
  route-count-aware pack/select/accumulate kernel. Removing one padding memset
  is not sufficient to close the dense gap.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --warmup 10 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-nozero --experiment V0.241_grouped_adaptive_nozero_probe --output results/runs/v0_241_grouped_adaptive_nozero_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --warmup 10 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-nozero --experiment V0.241_grouped_adaptive_nozero_probe --output results/runs/v0_241_grouped_adaptive_nozero_seed17.json
```

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `tests/test_qwen_packed_dispatch.py`
- `results/runs/v0_241_grouped_adaptive_nozero_seed2026.json`
- `results/runs/v0_241_grouped_adaptive_nozero_seed17.json`
