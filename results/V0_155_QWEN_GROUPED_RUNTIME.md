# V0.155 Qwen Transferred-Neuron Runtime Audit

## Question

Does caching the transferred group weights make the grouped sparse bank faster,
and can the current PyTorch environment compile the dynamic dispatcher?

## Results

The grouped dispatcher now caches stacked slice weights as non-persistent bank
buffers instead of rebuilding them on every forward. A CPU numerical check
against the token-loop reference gives maximum absolute error `1.68e-8`.

On the stable 8-group/top-4, 8-layer, 50%-active quality configuration, the
buffered grouped full-model run is:

| parent mean ms | sparse mean ms | sparse/parent | quality |
|---:|---:|---:|:---:|
| 226.646 | 236.378 | 1.043x | pass (`+0.0129` CE delta) |

The cache removes repeated weight stacking but does not overcome the dynamic
sort, gather, padding, and small batched-matmul overhead. `torch.compile` was
also screened; compilation stopped because this environment has no working
Triton installation. No compiled speedup is claimed.

## Decision

**Keep the buffered grouped implementation for correctness, but do not claim
runtime acceleration yet.** A fused CUDA/Triton kernel or a different runtime
layout is required before the active-compute reduction becomes an end-to-end
latency win.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_buffered_grouped_calrank8_hard100_b8s128_seed2026.json`
