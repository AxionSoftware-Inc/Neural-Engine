# V0.238 — Shape-aware adaptive grouped runtime audit

**Date:** 2026-09-09  
**Status:** `ACCEPTED OPT-IN SHAPE POLICY; NOT GLOBAL DEFAULT`  
**Branch:** `exp/track-runtime`

## Question

The combined grouped path improves medium batches but pays cache and fused
operator overhead at single-row decode. This experiment adds a shape-aware
policy without changing route selection:

- one flattened token row: use the plain grouped path;
- more than one row: use the cached, prepacked, fused grouped path;
- correction fusion and the exact uniform K=5 accumulation remain enabled in
  both cases.

The policy changes only backend selection. It does not force a larger active
subset or alter the trained model.

## Protocol

Qwen3-0.6B, layers 19–26, `E=8`, `K=5`, rank 1, float32, and
`600/600/200` child/hard/router steps. Prefix length is 4. Each seed uses 20
CUDA-Graph warmups and 50 timing iterations at B1/B8/B32. The dense parent and
plain grouped path are measured in the same process. Eight-token fixed-shape
generation and full-model graph/eager parity are checked.

## Results

Graph time relative to the dense parent, in B1/B8/B32 order:

| Seed | Quality CE delta | Existing grouped | Adaptive grouped | Adaptive / grouped | Generation |
|---:|---:|:---|:---|:---|:---:|
| 2026 | `+0.043787` | `1.081x / 1.018x / 1.008x` | `1.072x / 0.983x / 0.997x` | `0.992x / 0.966x / 0.989x` | exact |
| 17 | `+0.037170` | `1.070x / 1.031x / 1.070x` | `1.071x / 1.015x / 1.051x` | `1.001x / 0.984x / 0.983x` | exact |

The policy removes the single-row cache penalty and keeps the medium-batch
benefit. Across the two seeds, it is consistently better than grouped at B8
and B32, although it is still slightly slower than dense in most cases. The
maximum graph/eager logit difference remains within the existing float32
range (about `1.6e-5`), and all generation token comparisons are exact.

The quality deltas remain inside the current `+0.05` gate. This is a runtime
result only; the backend policy does not improve the trained model’s quality.

## Decision

Accept `grouped-adaptive` as the best current opt-in runtime policy for the
tested decode shapes. Do not make it the global default yet: the remaining
1–7% dense gap, especially at B1, needs validation on longer prefixes and
prefill shapes, and the current rule is a simple row-count heuristic.

The next runtime experiment is a prefix/pre-fill shape sweep using this same
policy. After that, the meaningful kernel target remains route-count-aware
packing or a specialized grouped-GEMM backend; the adaptive policy is the
selection layer around that future backend.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 20 --iterations 50 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --experiment V0.238_grouped_adaptive_audit --include-grouped-adaptive --output results/runs/v0_238_grouped_adaptive_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --warmup 20 --iterations 50 --batch-sizes 1 8 32 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --experiment V0.238_grouped_adaptive_audit --include-grouped-adaptive --output results/runs/v0_238_grouped_adaptive_seed17.json
```

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `tests/test_qwen_packed_dispatch.py`
- `results/runs/v0_238_grouped_adaptive_seed2026.json`
- `results/runs/v0_238_grouped_adaptive_seed17.json`
