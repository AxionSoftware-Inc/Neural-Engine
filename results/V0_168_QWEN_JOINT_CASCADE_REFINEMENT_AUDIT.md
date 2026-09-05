# V0.168 — Joint Cascade Refinement Audit

## Question

V0.167 identified depth composition as the remaining failure: the learned
subset router is acceptable at four layers but the eight-layer cascade fails.
This audit tests whether a small logit-level joint refinement of all sparse
children can repair that cascade without changing the 70-class routing rule.

## Protocol

The run uses the same protocol as the V0.167 eight-layer control: Qwen3-0.6B,
float32 CUDA, layers `19--26`, 8 contiguous groups with top-4 active, rank-64
cross-group correction, subset-router supervision for 100 steps, 300 soft
child steps, and 100 hard steps at `1e-5`. The calibration and evaluation
corpora are explicitly `data/qwen_calibration.txt` and `data/qwen_eval.txt`.
After the per-layer training, all sparse children are jointly refined for 50
logit-distillation steps at learning rate `1e-5`, temperature `2` and batch
size `4`. The held-out gate is `+CE <= 0.05`.

## Results

| Variant | Held-out +CE | Top-1 | Gate |
|---|---:|---:|:---:|
| V0.167 eight-layer control, hard LR `1e-5` | `+0.5685` | 60.47% | FAIL |
| Joint cascade refinement, 50 steps, LR `1e-5` | `+0.9811` | 55.18% | FAIL |

The joint run's final refinement loss remained large (`76.29`). Its local
MSE also grew with depth, reaching `90.13` at layer 26 before the final
end-to-end evaluation.

## Interpretation

Jointly updating all correction modules does not repair the depth cascade. It
worsens the held-out CE delta by `0.4126` relative to the matched V0.167
control. The problem is therefore not solved by a generic end-to-end logit
objective; with the current parameterization, the optimizer can trade errors
between layers while the later sparse layers still receive a shifted hidden
state.

The exploratory run that omitted the explicit calibration/evaluation files is
not used as evidence because it did not match the established benchmark
corpus. Only the matched run above is a decision result.

## Decision

Reject generic joint cascade refinement as the next scaling path. Keep the
four-layer frozen subset-router configuration as the best learned endpoint and
do not move to 700M/1B yet. The next architecture test should constrain the
interface between consecutive sparse layers — for example, a per-layer
zero-start residual/error state or a teacher-state correction trained at the
actual sparse-prefix distribution — instead of jointly changing every child
with a single final-logit loss.

## Artifact

- `results/runs/qwen_multi_layer_crossgroup_8layers_e8k4_rank64_subsetrouter100_frozen_hard100_lr1e-5_joint50_lr1e-5_seed2026.json`
- `benchmark_qwen_multi_layer_transplant.py`
