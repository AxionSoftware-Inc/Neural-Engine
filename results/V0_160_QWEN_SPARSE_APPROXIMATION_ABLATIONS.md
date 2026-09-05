# V0.160 Qwen Sparse Approximation Ablations

## Question

After the held-out audit rejected the current group router, can a scalar
hard-route rescaling, a compact nonlinear correction cell, or a narrower late-
layer replacement recover the missing Qwen function without changing the
architecture more fundamentally?

## Protocol

All runs use the local Qwen3-0.6B checkpoint, float32 CUDA inference, seed
2026, the held-out corpus in `data/qwen_eval.txt`, and the same two late FFN
layers (25--26) unless noted. The child keeps 8 transferred groups and
selects 4 per token (50% of the transferred FFN body). `+CE` is the change in
held-out teacher-forced cross-entropy after the sparse path is fully active;
lower is better and the quality gate is `+0.05` or less.

## Results

| Variant | Active layers | Correction | Hard scale | Held-out +CE | Top-1 agreement | Parent timing |
|---|---:|---|---:|---:|---:|---:|
| Baseline low-rank | 2 | rank 8 | 1.0 | +0.2318 | 79.93% | 1.011x |
| Rescaled | 2 | rank 8 | 4.0 | +0.0931 | 79.83% | 1.009x |
| Over-rescaled | 2 | rank 8 | 6.0 | +0.1461 | 73.85% | 1.011x |
| Nonlinear macro-cell | 2 | SwiGLU rank 64 | E/K | +0.1502 | 83.11% | 1.013x |
| Wider nonlinear macro-cell | 2 | SwiGLU rank 384 | E/K | +0.1388 | 83.69% | 1.015x |
| Shared-basis correction | 2 | shared basis rank 64 | 4.0 | +0.1031 | 79.22% | 1.016x |
| Wider shared-basis correction | 2 | shared basis rank 384 | 4.0 | +0.0940 | 81.10% | 1.026x |
| Late-layer sparse path | 4 | rank 8 | E/K | +0.1819 | 78.08% | 1.023x |

The `scale=4` run is the best scalar-rescaling point, but it remains nearly
twice the gate tolerance. Increasing the scale to 6 reverses the improvement.
The nonlinear macro-cell gives a modest recovery, but even rank 384 does not
restore the held-out function and its copied buffers make storage larger than
the dense parent. The new shared-basis correction is slightly better at rank
384 (`+0.0940`) than at rank 64 (`+0.1031`), but it does not beat the simpler
scale-4 control in a meaningful way and increases storage/runtime. Restricting
replacement to late layers improves the error relative to the full eight-layer
cascade, but still fails the generalization gate.

## Decision

These experiments reject four local repairs as sufficient solutions:

1. A fixed hard-route scale cannot compensate for missing cross-group
   interactions.
2. A small nonlinear residual cell improves the approximation only marginally;
   it does not preserve the parent FFN function at 50% active body compute.
3. Reducing the number of replaced layers hides some error but does not solve
   the architecture, and it also reduces the possible compute savings.
4. Giving selected groups a shared nonlinear basis does not recover the
   omitted function at acceptable storage or runtime.

The evidence now points to a structural decomposition problem: the current
groups are independent slices of Qwen's SwiGLU intermediate dimension, while
the parent output is a sum of many interacting neuron contributions. The next
high-value path is a function-preserving micro-group design with teacher-
derived cross-group mixing and a compiled selected-neuron kernel. It should be
tested on the same held-out corpus before any 300M--1B scale run. A full
neuron-level training run is also deferred until its selected-neuron matmul is
fused; the current Python gather path is about 3x slower even on two layers.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_two_layer_transplant.py`
- `data/qwen_calibration.txt`
- `data/qwen_eval.txt`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_2layers_e8k4_scale1_readme_eval_hard50_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_2layers_e8k4_scale4_readme_eval_hard50_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_2layers_e8k4_scale6_readme_eval_hard50_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_2layers_e8k4_macro64_readme_eval_hard50_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_2layers_e8k4_macro384_readme_eval_hard50_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_2layers_e8k4_sharedbasis64_scale4_readme_eval_hard50_lr3e-4_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_2layers_e8k4_sharedbasis384_scale4_readme_eval_hard50_lr3e-4_b8s128_seed2026.json`
