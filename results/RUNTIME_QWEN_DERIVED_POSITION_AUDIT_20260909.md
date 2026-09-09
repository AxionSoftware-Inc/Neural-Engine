# V0.257 — Derived-position fixed-pack audit

**Date:** 2026-09-09  
**Status:** `REJECTED AS MATERIAL OPTIMIZATION; OPT-IN PROBE RETAINED`  
**Branch:** `exp/track-runtime`

## Question

The fixed-layout pack stores each selected hidden row at
`grouped[expert, token]`. The previous finalizer received an int64
`packed_positions` tensor containing those rows. This probe derives the row
inside the finalizer from `top_ids[token, slot]` and the token index, removing
that metadata tensor and its construction from the fixed-pack path.

No router, route IDs, expert weights, grouped GEMM, correction, accumulation
order, or model parameter changes were made.

## Protocol

Local `Qwen/Qwen3-0.6B`, layers `19--26`, `E=8`, `K=5`, rank-1 correction,
float32 CUDA, `matmul_precision=highest`. The trained-cascade recipe was
`child_steps=600`, `hard_steps=600`, `router_steps=200`; prefix length 4;
batch sizes 1, 8, and 32; 20 warmups and 50 timing iterations. The control is
`grouped-adaptive-fixed-pack`; the probe is
`grouped-adaptive-fixed-pack-derived-position`. Seeds were 2026 and 17.

## Full-cascade latency

Lower is better. Percentages are derived-position minus fixed-pack relative to
the control.

| batch | fixed eager mean | derived eager mean | eager change | fixed graph mean | derived graph mean | graph change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38.3863 ms | 35.3641 ms | −7.74% | 10.7880 ms | 10.6956 ms | −0.86% |
| 8 | 33.6990 ms | 32.6470 ms | −3.12% | 16.5639 ms | 16.5683 ms | +0.03% |
| 32 | 38.0775 ms | 38.0382 ms | −0.10% | 33.9467 ms | 33.9628 ms | +0.05% |

The eager B=1/B=8 improvements are consistent with avoiding the metadata
construction, but the CUDA Graph serving path is effectively unchanged. The
remaining projection/finalization work dominates after capture.

## Correctness and quality

- Both seeds passed the strict graph/eager numerical gate; the maximum error
  was `1.31e-5` against the `1e-3` tolerance.
- Both seeds produced exact 8-token greedy-generation matches against the
  grouped baseline.
- The full model and training recipe are unchanged, so this is a runtime-only
  result. The reported dense-teacher CE deltas were `+0.044949` (seed 2026)
  and `+0.038644` (seed 17).

## Decision

Keep the derived-position finalizer as an opt-in implementation and do not
make it the default. It removes an allocation and gives a real eager benefit,
but it does not produce a material graph-serving gain. Further Python-level
metadata trimming is now low priority; a meaningful runtime result requires a
fused projection/finalization kernel that preserves batched GEMM efficiency or
a change that improves the quality/capacity bottleneck rather than another
small dispatch optimization.

## Files and rerun

- `neural_engine/qwen_grouped_finalize.cu`
- `neural_engine/qwen_grouped_finalize.cpp`
- `neural_engine/qwen_grouped_finalize.py`
- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `benchmark_qwen_grouped_stage_profile.py`
- `results/runs/v0_257_derived_position_screen_seed2026.json`
- `results/runs/v0_257_derived_position_long_seed17.json`

```powershell
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --warmup 20 --iterations 50 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-fixed-pack --include-grouped-adaptive-fixed-pack-derived-position --experiment V0.257_derived_position_screen --output results/runs/v0_257_derived_position_screen_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --warmup 20 --iterations 50 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-fixed-pack --include-grouped-adaptive-fixed-pack-derived-position --experiment V0.257_derived_position_long --output results/runs/v0_257_derived_position_long_seed17.json
```
