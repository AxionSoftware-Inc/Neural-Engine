# V0.255 — Vectorized fixed-pack audit

**Date:** 2026-09-09  
**Status:** `REJECTED AS MATERIAL OPTIMIZATION; OPT-IN PROBE RETAINED`  
**Branch:** `exp/track-runtime`

## Question

V0.250's fixed-layout pack uses one CUDA block per selected token/slot pair,
with scalar float32 copies across the hidden dimension. This probe adds a
second kernel that copies four adjacent floats per thread with `float4` loads
and stores. It changes no routes, tensor layout, grouped-GEMM shape, model
weights, correction math, or accumulation order.

The full path uses the scalar kernel at B=1 because the isolated probe showed a
decode penalty there, and uses the vectorized kernel only when the flattened
token count is greater than one.

## Isolated pack probe

Qwen hidden size was 1024, `E=8`, `K=5`, float32 CUDA, 200 timed iterations
after 30 warmups. Only selected expert/token rows are semantically compared;
unused fixed-layout rows are intentionally uninitialized in both kernels.

| flattened tokens | scalar pack | float4 pack | change |
|---:|---:|---:|---:|
| 1 | 0.01516 ms | 0.02338 ms | +54.2% |
| 8 | 0.03008 ms | 0.01444 ms | −52.0% |
| 32 | 0.03470 ms | 0.02145 ms | −38.2% |

Selected-row max error was `0.0` for every tested shape.

## Full-model protocol

Local `Qwen/Qwen3-0.6B`, layers `19--26`, `E=8`, `K=5`, rank-1 correction,
float32 CUDA with `matmul_precision=highest`. The trained-cascade recipe was
`child_steps=600`, `hard_steps=600`, `router_steps=200`; prefix length 4;
batch sizes 1, 8, and 32; 20 warmups and 50 timing iterations. The control is
V0.250 `grouped-adaptive-fixed-pack`; the probe is
`grouped-adaptive-fixed-pack-vectorized`. Seeds were 2026 and 17.

## Full-cascade latency

Lower is better. Percentages are vectorized minus scalar relative to the
fixed-pack control.

| batch | fixed eager mean | vectorized eager mean | eager change | fixed graph mean | vectorized graph mean | graph change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36.0378 ms | 35.7110 ms | −0.91% | 10.7287 ms | 10.7088 ms | −0.19% |
| 8 | 33.2006 ms | 32.7849 ms | −1.25% | 16.5460 ms | 16.5376 ms | −0.05% |
| 32 | 37.8935 ms | 37.9478 ms | +0.14% | 33.7730 ms | 33.7946 ms | +0.06% |

The per-shape seed changes are not consistent: B=8 improves in seed 17 but
slightly regresses in seed 2026; B=32 is effectively tied. The small eager
mean at B=8 does not survive in CUDA Graph serving latency.

## Correctness and quality

- Both seeds passed the strict graph/eager numerical gate; maximum error was
  `1.24e-5` against the `1e-3` tolerance.
- Both seeds produced exact 8-token greedy-generation matches against the
  grouped baseline.
- The quality recipe was unchanged; CE deltas versus the dense teacher were
  `+0.042170` (seed 2026) and `+0.037269` (seed 17). This is not a quality
  experiment.

## Decision

Do not make the vectorized pack the default and do not treat it as a capacity
or quality result. The implementation remains available as an opt-in
benchmark path because it is parity-safe and harmless, but pack memory
instruction tuning is no longer the main runtime target. The next useful
optimization must reduce a larger graph-visible stage or preserve more
cuBLAS/GEMM reuse; micro-optimizing this pack boundary is below the measured
noise floor.

## Files and rerun

- `neural_engine/qwen_deterministic_pack.cu`
- `neural_engine/qwen_deterministic_pack.cpp`
- `neural_engine/qwen_deterministic_pack.py`
- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `results/runs/v0_255_vectorized_pack_long_seed2026.json`
- `results/runs/v0_255_vectorized_pack_long_seed17.json`

```powershell
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --warmup 20 --iterations 50 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-fixed-pack --include-grouped-adaptive-fixed-pack-vectorized --experiment V0.255_vectorized_pack_long --output results/runs/v0_255_vectorized_pack_long_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --warmup 20 --iterations 50 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-fixed-pack --include-grouped-adaptive-fixed-pack-vectorized --experiment V0.255_vectorized_pack_long --output results/runs/v0_255_vectorized_pack_long_seed17.json
```
