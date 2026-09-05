# V0.158 Qwen Sparse Routing Held-Out Audit

## Question

Does the transferred-neuron group bank preserve Qwen quality on text that is
not used for calibration, and can partitioning or finer groups repair the
observed degradation?

## Protocol

The teacher is the frozen local Qwen3-0.6B model. Layers 19--26 are replaced
sequentially with copied Qwen SwiGLU slices. The standard runs use grouped
hard dispatch, 300 local calibration steps, 50 final hard-route steps, rank-8
base-output correction, batch 8 x 128, and seed 2026. `data/qwen_eval.txt` is
held out from calibration. The historical synthetic evaluation remains
reported as a control only.

## Results

| layout / active budget | calibration | evaluation | alpha=0 CE delta | teacher top-1 | gate |
|---|---|---|---:|---:|:---:|
| 8 groups / top-4 (50%) | diverse corpus | synthetic control | +0.0309 | 91.43% | pass |
| 8 groups / top-4 (50%) | diverse corpus | held-out corpus | +0.2919 | 59.84% | fail |
| 8 groups / top-4 (50%) | README | held-out corpus | +0.3217 | 60.01% | fail |
| 8 groups / top-4 (50%) interleaved | README | held-out corpus | +0.4399 | 62.40% | fail |
| 8 groups / top-4 (50%) norm-balanced | README | held-out corpus | +0.4786 | 58.98% | fail |
| 32 groups / top-8 (25%) | README | held-out corpus | +0.5346 | 55.54% | fail |
| 32 groups / top-16 (50%) | README | held-out corpus | +1.4395 | 42.14% | fail |
| 8 groups / top-4 oracle-dot* | none | held-out corpus | +0.7194 | 50.34% | fail |

The 50%-active synthetic pass therefore does not generalize to the held-out
corpus. Prompt parity also shows repetitive or incomplete continuations,
although the base Qwen checkpoint itself is not instruction-tuned and is not
expected to answer every prompt perfectly.

\* `oracle-dot` is a diagnostic route that evaluates all group outputs to
choose a subset, so it is not a speed result. Its failure indicates that the
current group approximation and rescaling are themselves a major bottleneck;
the learned router is not the only problem.

## Additional controls

- Removing hard-route training caused the 25%-active synthetic delta to jump
  to `+1.0448`; soft-to-hard mismatch is real, but fixing it is insufficient.
- Separate hard learning rate `3e-4` improved the diverse 25%-active synthetic
  result from `+0.0844` to `+0.0587`, still outside the `+0.05` gate.
- Rank-64 correction (`+0.0677`) and input-conditioned rank-64 correction
  (`+0.0831`) did not repair the 25%-active result.
- 75% active (`8/top-6`) also failed the synthetic gate at `+0.0762`; more
  active groups are not monotonically better under the current training path.

## Decision

**Reject the current group-routing claim as a general language-model
replacement.** Qwen neuron transfer remains a valid exact dense control, but
the current sparse group path is only a synthetic local approximation. Do not
scale this path to 1B parameters yet.

The next architecture experiment should change the approximation itself: use
an implementation that can select individual neurons or a learned
functionally coherent micro-group, with a held-out language evaluation in the
gate from the start. Static interleaving and norm balancing are rejected.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `data/qwen_calibration.txt`
- `data/qwen_eval.txt`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_diverse_eval_hard50_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_readme_eval_hard50_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_interleaved_readme_eval_hard50_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_normbalanced_readme_eval_hard50_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e32k8_readme_eval_hard50_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e32k16_readme_eval_hard50_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_oracle_dot_eval_nosteps_b8s128_seed2026.json`
