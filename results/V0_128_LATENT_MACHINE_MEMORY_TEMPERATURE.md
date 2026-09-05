# V0.128 `taklif22` Hierarchical Memory Temperature

## Question

Can the 512-entity hierarchical memory bottleneck be fixed by sharpening or
softening the two-stage address distribution?

## Protocol

All runs use 512 entities, 32 buckets (16 rows per bucket), task-only learned
memory, bounded two-dimensional state, four top-1 MLP cells, and the same
labels-free route objective. Only the memory temperature changes. The
temperature is applied to both bucket and within-bucket row logits.

## Results

| temperature | seed | depth-6 MSE | fact-table B MSE | memory accuracy | in-range MSE | decision |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.5 | 2026 | **6.06e-5** | 1.25e-4 | 87.6% | 1.04e-3 | mixed |
| 1.0 | 2026 | 7.31e-5 | 1.39e-4 | **92.3%** | **7.34e-4** | baseline |
| 2.0 | 2026 | 4.25e-4 | 5.32e-4 | 89.5% | 1.75e-3 | reject |

Temperature `0.5` slightly improves the fixed depth-6 metric but lowers exact
retrieval and does not improve in-range quality. Temperature `2.0` makes the
memory read too diffuse and degrades both retrieval and task error. The
tradeoff confirms that task quality and exact memory addressing are distinct
metrics.

## Decision

**Keep temperature 1.0. Reject 2.0; do not promote 0.5 as the default.** The
remaining 512-row bottleneck needs a better address objective, such as local
sampled negatives or a contrastive memory loss, rather than temperature-only
tuning.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_hierarchical32_temp05_memory_nosup_512e_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_hierarchical32_temp20_memory_nosup_512e_seed2026.json`
