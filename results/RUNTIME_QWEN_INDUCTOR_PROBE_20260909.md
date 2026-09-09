# V0.207 — Inductor fused-child probe

**Date:** 2026-09-09  
**Status:** `ENVIRONMENT-BLOCKED; DEFAULT UNCHANGED`  
**Branch:** `exp/track-runtime`

## Question

Before writing a custom CUDA kernel, the trained sparse child was passed to
PyTorch Inductor with `max-autotune-no-cudagraphs`. The goal was to see whether
routing, projection, and pointwise work could be fused without changing the
model or default execution path.

## Result

The probe did not compile. The installed PyTorch (`2.6.0+cu124`) reported:

```text
RuntimeError: Cannot find a working triton installation. Either the package is not installed or it is too old.
```

The ordinary eager and CUDA Graph paths still completed with `PARITY_PASS`;
this is an environment/toolchain limitation, not evidence against the sparse
architecture. The BMM and einsum A/B probes remained numerically correct.

## Decision

- keep the normal einsum/BMM-correction runtime unchanged;
- do not install or alter the environment as part of the model benchmark;
- leave the Inductor path as an opt-in probe for a machine with a compatible
  Triton installation;
- continue with a real static-index/fused correction implementation that does
  not depend on Inductor availability.

The benchmark also records the full compiler error and keeps the known failing
packed-capture probe last, so a failed optional backend cannot corrupt later
measurements.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 8 --seed 2026 --warmup 20 --iterations 20 --correction-backend-iterations 15 --single-token-backend-iterations 20 --compiled-child-iterations 50 --batch-sizes 1 8 --output results/runs/v0_207_inductor_rank8.json
```

## Artifact

- opt-in probe: `benchmark_qwen_trained_graph_audit.py`;
- sparse child and BMM correction: `benchmark_qwen_multi_layer_transplant.py`.
