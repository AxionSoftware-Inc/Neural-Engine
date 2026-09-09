# V0.206 — Single-token projection backend audit

**Date:** 2026-09-09  
**Status:** `BMM REJECTED AS DEFAULT; EINSUM RETAINED`  
**Branch:** `exp/track-runtime`

## Question

The sparse child’s single-token base path used three `einsum` contractions.
This audit compares an opt-in BMM implementation with the existing einsum
path on the same trained rank-8 K=5 cascade. Correction BMM is unchanged; only
the base child projection backend is switched.

## Trained B8 result

| backend | eager | graph | graph / eager | max graph parity |
|---|---:|---:|---:|---:|
| einsum | `37.626 ms` | `34.974 ms` | `0.930x` | `1.05e-5` |
| BMM | `37.238 ms` | `35.494 ms` | `0.953x` | `8.58e-6` |

BMM is about 1% faster in eager mode but about 1.5% slower in graph replay.
The full trained graph remains numerically correct, and the unit test passes,
but there is no reliable end-to-end runtime gain. The BMM base path is
therefore rejected as the default; the implementation remains available for
future hardware-specific testing. The default remains `einsum`.

## Audit reliability fix

The trained audit intentionally probes a packed correction backend that cannot
be captured because of device-side `torch.where`. A capture failure can poison
the current CUDA context, so the audit now runs all later A/B probes first and
keeps the intentionally failing packed probe last. This is a benchmark
correctness fix, not a model-quality change.

## Decision

- keep `single_token_projection_backend="einsum"` as the default;
- do not claim BMM as a speed improvement;
- retain the BMM path and parity test as an opt-in backend;
- continue toward a fused/static-index correction kernel for the remaining B8
  bottleneck.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 8 --seed 2026 --warmup 40 --iterations 60 --correction-backend-iterations 30 --single-token-backend-iterations 100 --batch-sizes 1 8 --output results/runs/v0_206_single_token_bmm_rank8.json
```

## Artifact

- projection implementation and default: `benchmark_qwen_multi_layer_transplant.py`;
- trained A/B harness: `benchmark_qwen_trained_graph_audit.py`;
- parity coverage: `tests/test_qwen_packed_dispatch.py`.
