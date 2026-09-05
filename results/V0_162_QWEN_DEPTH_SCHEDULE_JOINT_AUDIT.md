# V0.162 Qwen Depth Schedule and Joint Cascade Audit

## Question

The cross-group child passes on four late layers but fails when eight layers
are replaced. Can a depth-aware active schedule or a leakage-free joint
refinement remove the cascade error without changing the local sparse cell?

## Protocol

Runs use the local Qwen3-0.6B checkpoint, float32 CUDA, diversified
calibration from `data/qwen_calibration.txt`, held-out evaluation from
`data/qwen_eval.txt`, cross-group rank-64 correction, and 300 sequential child
steps with 200 hard-route steps. The held-out gate is `+0.05` CE delta or
lower. Joint refinement, when enabled, uses calibration teacher logits only;
the held-out logits remain evaluation-only.

## Results

| Configuration | Active schedule | Extra training | Held-out +CE | Top-1 | Timing |
|---|---|---|---:|---:|---:|
| 8 layers, uniform | 50% (`K4/8`) | none | +0.0854 | 65.58% | 1.646x |
| 8 layers, uniform | 75% (`K6/8`) | none | +0.0777 | 66.60% | 2.087x |
| 8 layers, early-heavy | 75% early, 50% late | none | +0.0861 | 66.02% | 1.878x |
| 8 layers, late-heavy | 50% early, 75% late | none | +0.0869 | 66.31% | 3.894x |
| 8 layers, uniform | 50% (`K4/8`) | joint, 4 calibration batches | +0.0953 | 66.85% | 1.648x |
| 8 layers, uniform | 50% (`K4/8`) | joint, 16 calibration batches | +0.0769 | 68.04% | 1.727x |

For reference, the same cross-group cell on four layers and 50% active passes
on two seeds at `+0.0181` and `+0.0155`. The schedule parser now supports one
active count and one hard-route scale per replaced layer. Joint refinement has
an independent small batch-size option so it can run without the previous GPU
out-of-memory failure.

## Decision

The 8-layer failure is not fixed by simply allocating more active groups to
every layer, putting the extra budget early or late, or jointly refining on a
small calibration subset. The 16-batch joint run is the best of these
follow-ups but still misses the gate by `0.0269` CE. It also adds training
complexity without changing the fundamental sparse decomposition.

Keep the 4-layer 50%-active result as a conditional local proof. Do not call
the method a full-stack Qwen replacement and do not scale it to 1B yet. The
next research branch must change the representation of the sparse child, for
example a teacher-derived shared residual factored across depth, or a
different function decomposition that does not sum independent group slices.
Only after that passes the same held-out gate should we spend time on a fused
kernel and larger models.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_crossgroup64_scale4_diverse_eval_hard200_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k6_crossgroup64_scale2p666_diverse_eval_hard200_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8_schedule6-4_crossgroup64_diverse_eval_hard200_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8_schedule4-6_crossgroup64_diverse_eval_hard200_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_crossgroup64_scale4_diverse_eval_hard200_joint100_calbatch1_lr1e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_crossgroup64_scale4_diverse_eval_hard200_joint100_cal16_batch1_lr1e-4_b8s128_seed2026.json`
