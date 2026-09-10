# P-003 native CUDA Graph audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Device: NVIDIA GeForce RTX 3060, CUDA 12.4

## Question

Can the native fixed-width and learned dynamic-width serving paths use a
static CUDA Graph without changing outputs or freezing an unsafe route pattern?

## Results

| Variant | Batch | Status | Graph/eager | Max logit error |
|---|---:|---|---:|---:|
| Fixed K=16 | 1 | OK | 1.006x | 0.0 |
| Learned K=8/16 with small-batch guard | 1 | OK | 1.001x | 0.0 |
| Learned K=8/16 dynamic dispatch | 32 | **CAPTURE FAILED** | — | — |

The batch-1 learned case uses the configured `dynamic_width_min_batch=32`
fallback, so it is effectively a fixed K=16 graph. It preserves parity but
does not provide a meaningful graph speedup over warmed eager execution.

At batch 32 the dynamic path reaches a CUDA Graph capture error because the
route-dependent grouped dispatch uses GPU-derived conditions and variable
narrow/wide subsets during capture (`operation not permitted when stream is
capturing`). A graph cannot safely replay a route partition captured from one
input batch for a different batch.

## Decision

`OPEN BLOCKER — DYNAMIC ROUTE IS NOT GRAPH-SAFE; FIXED FALLBACK ONLY`.

The fixed path is parity-safe but not a serving breakthrough. The learned path
remains eager/opt-in for batches above the guard threshold. A future graph
implementation must use a stable dispatch shape or a custom static-index
kernel that preserves route-dependent outputs; it must not silently reuse a
captured partition.

## Raw evidence and reproduction

- [Fixed K=16 batch-1 JSON](diagnostic_native_cuda_graph_fixed16_b1_20260910.json)
- [Learned batch-1 JSON](diagnostic_native_cuda_graph_learned_b1_20260910.json)
- [Learned batch-32 JSON](diagnostic_native_cuda_graph_learned_b32_20260910.json)
- [CUDA Graph benchmark](../benchmark_native_cuda_graph.py)

```powershell
python benchmark_native_cuda_graph.py `
  --checkpoint results/checkpoints/ne500_stable_prefix_active16_s17_3000.pt `
  --batch-size 1 --iterations 100 `
  --output results/diagnostic_native_cuda_graph_fixed16_b1_20260910.json

python benchmark_native_cuda_graph.py `
  --checkpoint results/checkpoints/ne500_stable_prefix_active16_s17_3000_learned_width.pt `
  --batch-size 32 --iterations 50 `
  --output results/diagnostic_native_cuda_graph_learned_b32_20260910.json
```
