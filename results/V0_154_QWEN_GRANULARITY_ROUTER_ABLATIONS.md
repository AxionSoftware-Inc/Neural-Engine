# V0.154 Qwen Neuron-Group Granularity and Router Ablations

## Question

Can finer copied-neuron groups make the 25%-active operating point stable,
and can a contribution-based router objective improve selection?

## Results

The 8-layer cascade uses the transferred Qwen SwiGLU slices and rank-8
calibration.

| bank | seed | active fraction | alpha=0 CE delta | teacher top-1 | gate |
|---|---:|---:|---:|---:|:---:|
| 8 groups/top-2 | 2026 | 25% | +1.3736 | 69.21% | fail |
| 16 groups/top-4 | 2026 | 25% | +0.0499 | 91.41% | pass |
| 16 groups/top-4 | 2027 | 25% | +0.0547 | 90.53% | fail |

Splitting the parent intermediate dimension into twice as many groups is a
large improvement, but the two-seed result is still just outside the quality
gate and cannot be promoted as stable.

An importance-router ablation trained the cheap router against the norm of
each frozen group contribution, without child MSE training. At 16 groups/top-4
it reaches `+0.3942` CE delta and `86.65%` agreement. The contribution norm is
not a sufficient target for reconstructing the signed full FFN output.

## Decision

**Keep 8 groups/top-4 (50% active) as the stable quality default.** Reject
8 groups/top-2 and the contribution-oracle router. Keep 16 groups/top-4 as a
promising but unvalidated 25%-active branch; it needs a better signed-output
reconstruction or interface correction before use.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k2_grouped_calrank8_hard100_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e16k4_grouped_calrank8_hard100_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e16k4_grouped_calrank8_hard100_b8s128_seed2027.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e16k4_routeroracle300_b8s128_seed2026.json`
