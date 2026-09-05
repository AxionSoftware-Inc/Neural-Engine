# V0.161 Qwen Cross-Group Mixing Scale Study

## Question

Can a selected Qwen group learn to compensate for omitted group interactions
when its own transferred output is passed through a teacher-trained low-rank
cross-group map? If so, how far does that result scale in depth and active
fraction?

## Method

The base child copies Qwen3-0.6B SwiGLU neurons into contiguous groups and
selects groups with a learned router. The new cross-group correction takes the
actual output of each selected group, projects it through a small latent map,
and projects it back to the hidden state. The correction is zero-start on the
output side, so the all-groups soft path remains an exact transfer control.

All quality numbers below use the same held-out corpus in
`data/qwen_eval.txt`, float32 CUDA inference, 8 groups, rank-64 correction,
diversified calibration from `data/qwen_calibration.txt`, and 300 total child
steps with 200 hard-route steps at hard learning rate `3e-4`. `+CE` is the
fully active sparse-vs-teacher held-out cross-entropy delta; the gate is
`+0.05` or lower.

## Results

| Configuration | Active body | Seed | Held-out +CE | Teacher top-1 | End-to-end timing |
|---|---:|---:|---:|---:|---:|
| 2 layers (25--26), K=4/8 | 50% | 2026 | +0.0306 | 85.06% | 1.159x |
| 2 layers (25--26), K=4/8 | 50% | 2027 | +0.0306 | 84.64% | 1.159x |
| 4 layers (23--26), K=4/8 | 50% | 2026 | +0.0181 | 80.47% | 1.324x |
| 4 layers (23--26), K=4/8 | 50% | 2027 | +0.0155 | 80.42% | 1.321x |
| 8 layers (19--26), K=4/8 | 50% | 2026 | +0.0854 | 65.58% | 1.646x |
| 8 layers (19--26), K=6/8 | 75% | 2026 | +0.0777 | 66.60% | 2.087x |
| 4 layers (23--26), E=16/K=4 | 25% | 2026 | +0.1282 | 73.78% | 1.338x |

The 2-layer and 4-layer results pass the held-out gate on both tested seeds.
This is the first repeatable positive signal after the earlier group-routing
failures. The 8-layer result fails at 50% active, and increasing to 75% only
reduces the error; it does not restore the function. Reducing the active body
to 25% fails even at four layers.

## Interpretation

The cross-group map repairs a real part of the sparse approximation error. Its
benefit is not explained by random initialization: the two-seed results are
nearly identical. However, the result is conditional rather than a general
replacement for Qwen's FFN:

- The depth boundary is currently between four and eight replaced layers.
- The active-fraction boundary is currently between 50% and 25% at four layers.
- The Python grouped implementation is slower end-to-end, and copied Qwen
  buffers make reported child storage about `2.125x` the parent. This is a
  quality result, not yet a deployment-speed or memory result.

## Decision and next step

Keep cross-group output mixing as the current research direction. Do not move
to 300M--1B scale yet, and do not claim that 50% sparsity works for a full
Qwen stack. The next experiment is a depth-aware schedule and leakage-free
joint cascade refinement. Those follow-up results are recorded in
`V0_162_QWEN_DEPTH_SCHEDULE_JOINT_AUDIT.md`. In parallel, replace the Python
gather path with a compiled selected-group kernel before making runtime claims.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_2layers_e8k4_crossgroup64_scale4_diverse_eval_hard200_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_2layers_e8k4_crossgroup64_scale4_diverse_eval_hard200_lr3e-4_b8s128_seed2027.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_4layers_e8k4_crossgroup64_scale4_diverse_eval_hard200_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_4layers_e8k4_crossgroup64_scale4_diverse_eval_hard200_lr3e-4_b8s128_seed2027.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_crossgroup64_scale4_diverse_eval_hard200_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k6_crossgroup64_scale2p666_diverse_eval_hard200_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_4layers_e16k4_crossgroup64_scale8_diverse_eval_hard200_lr3e-4_b8s128_seed2026.json`
