# V0.222 — Trained K=5 fused subset-router audit

**Date:** 2026-09-09  
**Status:** `PARITY-SAFE OPT-IN; DEFAULT UNCHANGED`  
**Branch:** `exp/track-runtime`

## Question

The real accepted Qwen K=5 recipe uses `route_source="subset-router"`, not the
ordinary 8-group router. Its router predicts 56 possible 5-group subsets.
V0.221 fused the subset MLP, best-subset argmax, and membership mapping. This
run checks that path on an actually trained eight-layer K=5 child using the
accepted `300/300/100`, rank-64 recipe.

Only inference routing changes between the two arms. Child weights, training
data, correction, K=5, and model body are unchanged.

## Quality control

The trained sparse child produced:

- teacher CE: `4.785308`;
- sparse CE: `4.827444`;
- CE delta: `+0.042135` (inside the existing `+0.05` K=5 gate);
- top-1 agreement: `0.8052`.

This is a one-seed backend equivalence run; the two-seed quality acceptance of
the K=5 recipe remains the existing baseline. The fused route is inference
only and does not retrain or alter quality targets.

## Runtime and parity

The measurement uses fixed custom KV cache and one-token decode, with 15
timing iterations after 10 warmups:

| Backend | Batch | Eager | CUDA Graph | Graph/eager logit error |
|---|---:|---:|---:|---:|
| PyTorch subset router | 1 | 33.7165 ms | 12.0220 ms | `5.72e-6` |
| Fused subset router | 1 | 33.0550 ms | 12.4084 ms | `6.20e-6` |
| PyTorch subset router | 8 | 36.9960 ms | 32.0519 ms | `9.54e-6` |
| Fused subset router | 8 | 37.7498 ms | 32.4018 ms | `1.10e-5` |

Direct backend comparisons on identical decode inputs were:

- eager fused/PyTorch: `0.980x` at B1 and `1.020x` at B8;
- graph fused/PyTorch: `1.032x` at B1 and `1.011x` at B8;
- maximum fused-vs-PyTorch eager logit error: `5.72e-6` at B1 and
  `1.07e-5` at B8.

The initial graph attempt exposed a current-stream bug and produced an invalid
route ID. Binding the extension to `getCurrentCUDAStream()` fixed it; the
re-run passed graph replay without any device assertion. This stream fix is
part of the implementation, not a model-quality change.

## Decision

- Keep `single_token_router_backend="torch"` as the default.
- Keep `"cuda-fused-subset"` as an opt-in eager one-token micro-optimization.
- Do not claim a CUDA Graph speedup: graph replay already removes much of the
  launch overhead and the fused route is slightly slower in this run.
- The real remaining runtime target is selected FFN/correction dispatch,
  especially for trained B8 and larger batches.
- The subset-router kernel is now relevant to the real K=5 architecture, but
  two-seed long timing and production stream isolation remain open before any
  serving policy changes.
- Eight-token CUDA-Graph greedy generation produced an exact token match
  between the two backends.

## Reproduction

```text
python -u benchmark_qwen_trained_subset_router_backend.py --warmup 10 --iterations 15 --calibration-rank 64 --output results/runs/v0_222_trained_fused_subset_router_backend.json
```

## Artifacts

- `benchmark_qwen_trained_subset_router_backend.py`
- `neural_engine/qwen_router_dispatch.py`
- `neural_engine/qwen_router_dispatch.cpp`
- `neural_engine/qwen_router_dispatch.cu`
- integration hook: `benchmark_qwen_multi_layer_transplant.py`
