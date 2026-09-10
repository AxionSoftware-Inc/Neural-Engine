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

## Serving batch-shape sweep

The same three checkpoints were measured at balanced batch sizes 15, 120, 240,
480, and 960 (five warm-ups and 20 synchronized CUDA repeats). The table is
the three-seed mean; the percentage is the reduction relative to the matching
PyTorch path.

| Batch | Fixed K=16 torch → fused | Speed change | Learned torch → fused | Speed change |
|---:|---:|---:|---:|---:|
| 15 | 7.52 → 7.25 ms | −3.5% | 8.51 → 6.59 ms | −22.5% |
| 120 | 23.17 → 11.35 ms | **−51.0%** | 19.60 → 15.40 ms | −21.4% |
| 240 | 38.75 → 17.91 ms | **−53.8%** | 28.50 → 18.91 ms | −33.7% |
| 480 | 65.48 → 29.76 ms | **−54.6%** | 44.57 → 29.21 ms | −34.5% |
| 960 | 116.43 → 45.31 ms | **−61.1%** | 67.88 → 41.56 ms | −38.8% |

All fused cases stayed within `5.72e-6` maximum logit error of their PyTorch
reference. At batch 15 the learned-width guard correctly selected full K=16;
at batches 120–960 its mean active width was approximately 8.4–8.9. Thus the
kernel is useful for both fixed-width throughput and learned-width serving,
but the small-batch result is too close to launch noise to justify a universal
automatic dispatch policy.

## Unsupported-feature fallback validation

The bank now exposes one explicit eligibility predicate for the native kernel.
Nine CUDA tests verified that unsupported configurations do not call the
native extension and instead complete through the existing PyTorch path:
unordered slots, shared factor mix, query-conditioned mix, pair interaction,
factor product, hidden product, hidden gate, serial composition, and address
residual. This is a safety check, not an implementation of those features in
the fused kernel.

## Sequence-shape sweep

At balanced batch 120, three seeds were also tested with the same padded
examples truncated to sequence lengths 6, 8, 16, and 32. The three-seed mean
speed reductions were:

| Sequence length | Fixed K=16 | Learned K=8/16 |
---:|---:|---:|
| 6 | **40.7%** | **29.5%** |
| 8 | **48.9%** | **29.6%** |
| 16 | **50.2%** | **30.7%** |
| 32 | **50.9%** | **27.9%** |

Every fused sequence case remained within `5.72e-6` maximum logit error of
the matching PyTorch path. The model's configured `slot_count=5` was respected;
length 6 is the shortest valid serving shape for this checkpoint. This closes
the current batch/sequence parity sweep for the tested 500M configuration, but
it is not yet a production integration test with request-shape caching or
concurrent requests.

## Serving reuse and stream-safety smoke

For each of the three seeds, B=120 requests at sequence lengths 6 and 32 were
run repeatedly in alternating shape order and then concurrently on two CUDA
streams. Both fixed K=16 and learned K=8/16 fused paths stayed within
`5.72e-6` maximum logit error of their torch references in both tests. No
shape-switch or cross-stream output contamination was observed. This validates
the kernel's current read-only serving contract, but it does not yet provide a
request pool, shape-cache eviction policy, or a production server integration.

## Long quality control

The fused learned checkpoints were rerun through the 96-batch-per-condition
OOD audit and compared with the existing PyTorch learned control. Exact,
hard-task, active-width, and wide-width metrics were identical for all three
seeds and four conditions at reported precision. The maximum absolute CE
difference was `1.12e-8`. This confirms that the small floating-point ordering
difference from atomic accumulation did not change the recurrent route or the
reported quality metrics in this audit.

## Independent long quality control

To separate a kernel regression from checkpoint variance, the three learned
checkpoints were evaluated for 48 balanced batches per condition with both
`native_cuda_fused` and `torch` backends. This is 1,536 examples per task per
condition. Across all three seeds and all four conditions, exact-accuracy
difference was `0`; the largest CE difference was `1.61e-8`.

The fused three-seed means were uniform exact `77.56%`, combination-heldout
`77.76%`, low-edge `95.76%`, and high-edge `95.20%`. Seed 19 itself was much
weaker on the uniform/deep composition conditions than seeds 17/18, but the
same weakness appeared in its torch control exactly. It is therefore a
training/checkpoint-seed stability issue, not a native-kernel quality issue.

## CUDA Graph check

Fixed K=16 with the native kernel captured successfully at batch 1 and 32.
Graph/eager ratios were `0.968x` and `0.837x`, with max errors
`1.91e-6` and `3.81e-6`. Learned dynamic routing remains graph-unsafe because
its route partition is variable; this kernel does not hide that separate issue.

## Decision

`PROMISING OPT-IN — BATCH/SEQUENCE/FALLBACK/STREAM SMOKE VALIDATED; PRODUCTION INTEGRATION OPEN`.

Keep native fused dispatch opt-in and leave PyTorch as the default. Batch,
sequence-shape, fallback, and stream-safety checks now pass, but a production
request shape-cache/fallback integration and an independent longer quality run
remain before any default switch. The kernel must never silently approximate a
configuration it does not support.

## Raw evidence and reproduction

- [480-batch runtime JSON](diagnostic_native_fused_runtime_all3_480_20260910.json)
- [960-batch runtime JSON](diagnostic_native_fused_runtime_all3_960_20260910.json)
- [Batch-shape sweep JSON](diagnostic_native_fused_shape_sweep_all3_20260910.json)
- [Sequence-shape sweep JSON](diagnostic_native_fused_sequence_sweep_all3_20260910.json)
- [Serving reuse/stream smoke JSON](diagnostic_native_fused_serving_smoke_all3_20260910.json)
- [Seed17 fused runtime smoke JSON](diagnostic_native_fused_runtime_s17_480_20260910.json)
- [Long fused learned-width OOD JSON](diagnostic_native_fused_learned_ood_long96_20260910.json)
- [Independent long fused OOD JSON](diagnostic_native_fused_ood_long48_all3_20260910.json)
- [Matching long torch OOD control JSON](diagnostic_native_torch_ood_long48_all3_20260910.json)
- [Fused fixed K=16 Graph batch-1 JSON](diagnostic_native_cuda_graph_fused_fixed16_b1_20260910.json)
- [Fused fixed K=16 Graph batch-32 JSON](diagnostic_native_cuda_graph_fused_fixed16_b32_20260910.json)
- [Runtime benchmark](../benchmark_native_width_runtime.py)
- [CUDA Graph benchmark](../benchmark_native_cuda_graph.py)
- [Python wrapper](../neural_engine/native_fused_dispatch.py)
- [CUDA kernel](../neural_engine/native_fused_dispatch.cu)
- [Batch-shape sweep](../benchmark_native_fused_shape_sweep.py)
- [Sequence-shape sweep](../benchmark_native_fused_sequence_sweep.py)
- [Serving smoke](../benchmark_native_fused_serving_smoke.py)

```powershell
python benchmark_native_width_runtime.py `
  --checkpoints results/checkpoints/ne500_stable_prefix_active16_s17_3000.pt `
    results/checkpoints/ne500_stable_prefix_active16_s18_3000.pt `
    results/checkpoints/ne500_stable_prefix_active16_s19_3000.pt `
  --examples-per-task 64 --warmup 5 --repeats 20 --include-native-fused `
  --output results/diagnostic_native_fused_runtime_all3_960_20260910.json
```
