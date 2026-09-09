# V0.223 — Graph-safe grouped selected-FFN audit

**Date:** 2026-09-09  
**Status:** `PROMISING RUNTIME PATH; SERVING POLICY OPEN`
**Branch:** `exp/track-runtime`

## Question

The accepted K=5 Qwen child already had a single-token path for decode, but
the grouped selected-FFN path was not CUDA-Graph-safe because it used
`torch.bincount` and a host-side dynamic `max_count`. This audit replaces
that capture-time-only shape decision with a graph-stable upper bound: a token
can contribute at most once to each expert, so `max_count = flattened_token_count`
is safe during capture. Eager execution retains the tighter dynamic bound.

The experiment compares both paths on the same freshly trained eight-layer
K=5 subset-router child. Router, circuit weights, correction, and model body
are unchanged.

## Results

The accepted `300/300/100`, rank-64 recipe was trained independently with
seeds 2026 and 2027. The quality gate remained inside `+0.05` CE in both
runs:

| Seed | Teacher CE | Sparse CE | CE delta | Top-1 agreement |
|---:|---:|---:|---:|---:|
| 2026 | 4.785308 | 4.825062 | `+0.039754` | 0.8052 |
| 2027 | 4.785308 | 4.825906 | `+0.040598` | 0.8091 |

Each timing run used 10 warmups and 20 iterations for seed 2026, and 10
warmups and 30 iterations for seed 2027. Ratios are grouped divided by the
single-token path:

| Seed | Batch | Eager grouped/single | CUDA Graph grouped/single | Max logit error |
|---:|---:|---:|---:|---:|
| 2026 | 1 | `1.132x` | `0.963x` | `7.87e-6` |
| 2026 | 8 | `0.919x` | `0.589x` | `9.42e-6` |
| 2026 | 32 | `0.479x` | `0.423x` | `1.38e-5` |
| 2027 | 1 | `1.167x` | `0.953x` | `5.72e-6` |
| 2027 | 8 | `0.965x` | `0.589x` | `9.54e-6` |

The grouped graph path therefore reduced measured decode graph time by about
4–5% at B1, 41% at B8, and 58% at B32 in the runs where those batch sizes were
measured. Eight-token CUDA-Graph greedy generation produced an exact token
match for both seeds. The remaining small logit differences are floating-point
reduction-order differences and did not change the tested generation.

An additional seed-2026 sweep varied the cached prefix length at B1 and B8:

| Prefix length | B1 graph ratio | B8 graph ratio | Max logit error |
|---:|---:|---:|---:|
| 4 | `0.952x` | `0.593x` | `9.66e-6` |
| 32 | `0.950x` | `0.600x` | `9.06e-6` |
| 128 | `0.959x` | `0.663x` | `1.08e-5` |

The gain persists with longer context, although attention increasingly
dilutes the selected-FFN improvement.

## Decision

- The grouped path is now graph-safe for the tested fixed-shape decode path.
- This is the strongest runtime signal in the current track and targets the
  real selected-FFN bottleneck, unlike router-only fusion.
- The B32 extension is positive, but longer production-shape timing and a
  broader generation/evaluation audit are still required before a serving
  policy is changed.
- Keep the eager policy conservative: grouped is slower at B1 but modestly
  faster at B8. CUDA Graph serving is the promising use case.
- The static capture workspace grows with flattened token count, so prefill
  continues using the dynamic eager bound.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 10 --iterations 20 --batch-sizes 1 8 32 --calibration-rank 64 --seed 2026 --output results/runs/v0_223_trained_qwen_dispatch_path_audit_b32.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 10 --iterations 30 --calibration-rank 64 --seed 2027 --output results/runs/v0_223_trained_qwen_dispatch_path_audit_seed2027.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 8 --iterations 10 --batch-sizes 1 8 --prefix-lengths 4 32 128 --calibration-rank 64 --seed 2026 --output results/runs/v0_223_trained_qwen_dispatch_path_audit_prefixes.json
```

## Artifacts

- `benchmark_qwen_trained_dispatch_path_audit.py`
- integration change in `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/v0_223_trained_qwen_dispatch_path_audit_b32.json`
- `results/runs/v0_223_trained_qwen_dispatch_path_audit_seed2027.json`
- `results/runs/v0_223_trained_qwen_dispatch_path_audit_prefixes.json`
