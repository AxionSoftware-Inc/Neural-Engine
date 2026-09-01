# V0.21 CPU-RAM circuit cache and paging

`taklif.md` separates architecture semantics from memory placement and asks for
measurements of cache hit rate, host-to-device traffic, and GPU idle cost. V0.21
adds an inference-only `CircuitRowCache` prototype. The router and recurrent
controller remain on the GPU; the trained 100M circuit bank stays in CPU RAM,
and selected `(down, up, bias)` rows are fetched into an LRU GPU cache.

The benchmark pre-generates different balanced batches so routes are not
artificially identical across iterations. Each model forward can issue three
cache requests because the checkpoint has three internal recurrent steps.

## Reproduction

```powershell
python benchmark_paging.py --checkpoint results/checkpoints/ne100_v12_coverage_full.pt --device cuda --batch-size 128 --iterations 100 --warmup 5 --cache-sizes 0 512 2048 7552 --output results/runs/ne100_paging_100.json
```

Hardware was an RTX 3060 12 GB. The cache size is measured in circuit rows.
The full-cache value is 7,552 rows; the measured resident count can be lower
when a row was never selected by the benchmark.

## Results

| GPU cache rows | Cache hit rate | H2D traffic | Latency/batch | Throughput | Peak VRAM | Resident rows |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.00% | 6,988.9 MB | 30.43 ms | 4,206/s | 127 MiB | 0 |
| 512 | 10.55% | 6,251.6 MB | 30.26 ms | 4,231/s | 190 MiB | 512 |
| 2,048 | 38.30% | 4,312.5 MB | 25.99 ms | 4,924/s | 240 MiB | 2,048 |
| 7,552 | **97.61%** | **167.0 MB** | **13.59 ms** | **9,422/s** | 470 MiB | 7,512 |

The cache dramatically reduces host-to-device traffic once it can hold most of
the working circuit set. The full-cache run still trails the existing
GPU-resident NE-100 inference measurement of roughly 14.6k samples/s because
this prototype performs Python-level row reconstruction and does not overlap
transfers with compute. That gap is a systems optimization target, not an
architecture result.

## Decision

CPU-RAM paging is technically viable, but a tiny cache is not enough for the
current route distribution. Before using paging for a long 300M/500M training
run, implement:

1. pinned host memory and asynchronous H2D prefetch;
2. batched contiguous row packing instead of Python-level `stack` calls;
3. route-aware batch grouping and next-step prefetch;
4. cache hit-rate and GPU-idle instrumentation during training;
5. dirty-row writeback for the lazy optimizer.

The prototype changes no model weights and is inference-only. It establishes
the required measurement path; it is not yet a production offload trainer.
