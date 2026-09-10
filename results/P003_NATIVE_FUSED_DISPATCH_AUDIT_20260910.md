# P-003 native factorized fused-dispatch audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Device: NVIDIA GeForce RTX 3060, CUDA 12.4

## Question

Can the current factorized-additive native circuit bank be executed by one
custom CUDA dispatch per selected `(token, circuit)` pair, preserving the
route output while removing the PyTorch/einsum dispatch overhead?

## Scope and parity contract

The kernel implements the exact current ordered factorized-additive formula:

- two factor rows are mixed with the learned per-address coefficients;
- GELU is applied to the mixed down projection;
- the mixed up projection and bias are accumulated with route weights;
- the result is written to the same `[tokens, state_dim]` correction output.

It is inference-only and currently supports the 500M configuration’s ordered
two-slot factor bank with no pair/product/hidden-gate/address-residual terms.
Unsupported configurations remain on the PyTorch implementation.

## Three-seed runtime result

Each timing used five warm-ups and 20 synchronized CUDA repeats. `native_fused`
uses the custom kernel; the corresponding `torch` variant is the existing
PyTorch/einsum path. Parity is measured against the corresponding PyTorch
output on the same batch.

### Balanced batch 480

| Variant | PyTorch | Native fused | Speed change | Max error |
|---|---:|---:|---:|---:|
| Fixed K=8 | 37.20 ms | 22.03 ms | **−40.8%** | 5.72e-6 |
| Fixed K=16 | 58.38 ms | 35.32 ms | **−39.5%** | 5.72e-6 |
| Learned K=8/16 | 41.32 ms | 30.90 ms | **−25.2%** | 7.63e-6 |

### Balanced batch 960

| Variant | PyTorch | Native fused | Speed change | Max error |
|---|---:|---:|---:|---:|
| Fixed K=8 | 65.85 ms | 34.78 ms | **−47.2%** | 5.72e-6 |
| Fixed K=16 | 105.70 ms | 57.49 ms | **−45.6%** | 7.63e-6 |
| Learned K=8/16 | 68.32 ms | 50.28 ms | **−26.4%** | 7.63e-6 |

The learned selector keeps the same active widths and route choices; the
kernel changes execution only. The larger-batch result confirms that the
gain is not limited to launch noise at one shape.

## Long quality control

The fused learned checkpoints were rerun through the 96-batch-per-condition
OOD audit and compared with the existing PyTorch learned control. Exact,
hard-task, active-width, and wide-width metrics were identical for all three
seeds and four conditions at reported precision. The maximum absolute CE
difference was `1.12e-8`. This confirms that the small floating-point ordering
difference from atomic accumulation did not change the recurrent route or the
reported quality metrics in this audit.

## CUDA Graph check

Fixed K=16 with the native kernel captured successfully at batch 1 and 32.
Graph/eager ratios were `0.968x` and `0.837x`, with max errors
`1.91e-6` and `3.81e-6`. Learned dynamic routing remains graph-unsafe because
its route partition is variable; this kernel does not hide that separate issue.

## Decision

`PROMISING OPT-IN — PARITY PASSED; SHAPE/CAPABILITY VALIDATION OPEN`.

Keep native fused dispatch opt-in and leave PyTorch as the default. Before any
default switch, validate more batch/sequence shapes, an independent long
quality run, and fallback behavior for every unsupported factor-bank feature.
The kernel must never silently approximate a configuration it does not support.

## Raw evidence and reproduction

- [480-batch runtime JSON](diagnostic_native_fused_runtime_all3_480_20260910.json)
- [960-batch runtime JSON](diagnostic_native_fused_runtime_all3_960_20260910.json)
- [Seed17 fused runtime smoke JSON](diagnostic_native_fused_runtime_s17_480_20260910.json)
- [Long fused learned-width OOD JSON](diagnostic_native_fused_learned_ood_long96_20260910.json)
- [Fused fixed K=16 Graph batch-1 JSON](diagnostic_native_cuda_graph_fused_fixed16_b1_20260910.json)
- [Fused fixed K=16 Graph batch-32 JSON](diagnostic_native_cuda_graph_fused_fixed16_b32_20260910.json)
- [Runtime benchmark](../benchmark_native_width_runtime.py)
- [CUDA Graph benchmark](../benchmark_native_cuda_graph.py)
- [Python wrapper](../neural_engine/native_fused_dispatch.py)
- [CUDA kernel](../neural_engine/native_fused_dispatch.cu)

```powershell
python benchmark_native_width_runtime.py `
  --checkpoints results/checkpoints/ne500_stable_prefix_active16_s17_3000.pt `
    results/checkpoints/ne500_stable_prefix_active16_s18_3000.pt `
    results/checkpoints/ne500_stable_prefix_active16_s19_3000.pt `
  --examples-per-task 64 --warmup 5 --repeats 20 --include-native-fused `
  --output results/diagnostic_native_fused_runtime_all3_960_20260910.json
```
