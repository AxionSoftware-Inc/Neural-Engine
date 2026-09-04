# V0.102 Grouped Sparse Execution Microbenchmark

## Question

Can the current factorized circuit representation use GPU locality more
efficiently without changing model weights or routing semantics?

## Protocol

The benchmark compares the exact same serial computation in three layouts:

1. `reference_factorized_gather`: compose factor rows independently for every
   routed sample and slot, matching the current implementation.
2. `grouped_unique_route`: compose each unique virtual route once per slot and
   index the result back to the batch.
3. `materialized_contiguous_page`: materialize all virtual circuit matrices in
   a contiguous page once, then gather complete matrices during execution.

The bank has 1,408 virtual circuits, 38 factor rows, state width 384, rank 16,
batch size 512, eight serial slots, five warmups and 20 timed iterations on
the RTX 3060. Each candidate is checked against the reference; all reported
maximum absolute errors are zero.

```powershell
python -u benchmark_grouped_sparse.py --device cuda --warmup 5 --iterations 20 --output results/runs/benchmark_grouped_sparse.json
```

## Results

| route pool | unique IDs/slot | reference ms | grouped ms | contiguous page ms | page memory |
|---:|---:|---:|---:|---:|---:|
| 32 | 32.0 | 8.06 | **4.65** | **2.06** | 68.1 MB |
| 128 | 125.5 | 7.99 | **5.74** | **2.31** | 68.1 MB |
| 1,408 | 432.6 | 8.03 | 9.75 | **2.45** | 68.1 MB |

Grouped execution helps when routes have strong locality, but is slower than
the reference when almost every route is unique. The contiguous page is the
fastest in this small bank, but its memory cost scales with the full virtual
bank: the same representation would be roughly 1.1 GB per bank at 23,600
circuits and roughly 1.9 GB per bank at 39,300 circuits, before counting the
three operation-specific banks.

## Decision

This is a positive systems signal, not yet a model-quality signal. Keep the
quality reference unchanged. A future runtime should select grouped execution
only above a measured locality threshold and use a page/cache policy for hot
routes; blindly grouping all routes is rejected. A custom fused CUDA kernel is
not claimed yet because no custom kernel was implemented in this screen.

## Artifact

- `results/runs/benchmark_grouped_sparse.json`
