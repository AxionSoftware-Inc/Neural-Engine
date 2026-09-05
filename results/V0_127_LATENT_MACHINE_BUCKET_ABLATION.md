# V0.127 `taklif22` Hierarchical Bucket Ablation

## Question

Is the V0.126 hierarchical-memory gain monotonic as the number of buckets
increases, or is there an optimization sweet spot?

## Protocol

All runs use 512 entities, bounded two-dimensional state, task-only learned
memory, four top-1 MLP cells, and the V0.123 labels-free route objective.
Only the hierarchical bucket count changes. The 32-bucket result is the
two-seed V0.126 baseline; 64 and 128 buckets are 2026 screening runs.

## Results

| buckets | seed | local rows/bucket | depth-6 MSE | fact-table B MSE | memory accuracy | decision |
|---:|---:|---:|---:|---:|---:|:---:|
| 32 | 2026 | 16 | **7.31e-5** | 1.39e-4 | **92.3%** | best |
| 32 | 2027 | 16 | **7.36e-5** | 9.23e-5 | **94.9%** | best |
| 64 | 2026 | 8 | 1.92e-4 | 2.45e-4 | 89.6% | reject |
| 128 | 2026 | 4 | 3.72e-4 | 4.59e-4 | 79.8% | reject |

Route usage stays balanced in every run, so the degradation is a memory
addressing effect rather than computation-cell collapse. Increasing buckets
reduces local row competition but makes the learned bucket decision harder;
the two effects do not cancel monotonically.

## Decision

**Keep 32 buckets for the current 512-row prototype. Reject 64 and 128 as
default settings.** The next experiment should retain the 32-bucket geometry
and tune the address objective/temperature, then move to a content-addressed
memory that avoids row-wise learned embeddings entirely.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_hierarchical64_memory_nosup_512e_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_hierarchical128_memory_nosup_512e_seed2026.json`
