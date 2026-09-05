# V0.165 — Qwen Router, Oracle, and Activation-Cluster Audit

## Question

V0.164 left two possible explanations for the sparse quality loss: the cheap
group router might select the wrong cells, or the copied group decomposition
may be unable to approximate the full nonlinear FFN at 50% active compute.
This audit separates those cases and tests an activation-signature partition.

## Protocol

Runs use the local Qwen3-0.6B checkpoint, float32 CUDA inference, layers
23--26, 8 groups with top-4 active, the held-out corpus in
`data/qwen_eval.txt`, 300 child steps, and 200 hard-route steps at learning
rate `3e-4`, unless noted. The quality gate is fully-active sparse-vs-teacher
held-out `+CE <= +0.05`.

## Results

| Variant | Seed | Held-out +CE | Top-1 | Gate |
|---|---:|---:|---:|:---:|
| Activation-signature balanced cluster + cross-group | 2026 | `+0.0377` | 81.27% | PASS |
| Activation-signature balanced cluster + cross-group | 2027 | `+0.0518` | 80.98% | FAIL |
| Contiguous cross-group, no hard-route training | 2026 | `+0.2492` | 77.51% | FAIL |
| Contiguous cross-group + teacher-dot router supervision | 2026 | `+0.0274` | 80.15% | PASS |
| Contiguous cross-group + teacher-dot router supervision | 2027 | `+0.1148` | 73.10% | FAIL |
| Contiguous rank-64 cross-group, oracle-dot eval | 2026 | `+1.1787` | 50.29% | FAIL |
| Contiguous rank-0, oracle-dot eval | 2026 | `+0.9487` | 54.54% | FAIL |

## Interpretation

The activation-signature partition is a useful diagnostic but not a reliable
improvement: it passes one seed and misses the gate on the second, while the
contiguous cross-group reference passes both seeds at `+0.0181` and `+0.0155`.
Grouping neurons with similar activation traces does not make a selected cell
an adequate substitute for the omitted cells.

Removing hard-route training makes the result much worse even though the soft
distillation MSE reaches approximately `1e-6`. Therefore low soft-path loss is
not evidence that the hard sparse path will preserve the function.

Teacher-dot router supervision passes one seed but is not reproducible. More
importantly, the correction-free oracle-dot control fails badly. This is not an
ideal subset oracle: dot alignment is only a heuristic and can choose a poor
combination when group outputs cancel or interact. The rank-64 oracle run is
additionally affected by the mismatch between its teacher-trained correction
and oracle selection. The later exact best-subset oracle audit is therefore
required before drawing a decomposition conclusion.

## Decision

Reject activation clustering and teacher-dot supervision as primary paths, and
do not scale either to 300M--1B. The dot-oracle failure alone is not evidence
that the copied cell decomposition is impossible; it only rejects that routing
heuristic. The exact best-subset follow-up determines whether a learned router
can recover the available decomposition headroom.

The next architecture experiment is a residual coreset: keep the copied
selected micro-groups, but train an always-on compact residual representation
of the omitted FFN contribution and use signed, input-conditioned coefficients
for selected cells. It will be evaluated first on two late layers with the
same held-out gate; only a two-seed pass can justify moving to four layers.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_activation_cluster_crossgroup_4layers_e8k4_rank64_seed2026.json`
- `results/runs/qwen_multi_layer_activation_cluster_crossgroup_4layers_e8k4_rank64_seed2027.json`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_nohard_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_routerdot100_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_routerdot100_seed2027.json`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_oracledot_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank0_oracledot_seed2026.json`
