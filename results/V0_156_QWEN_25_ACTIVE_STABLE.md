# V0.156 Qwen Transferred-Neuron 25%-Active Stability

## Question

Can a finer transferred-neuron bank pass the eight-layer quality gate at only
25% active intermediate compute with a stable training schedule?

## Protocol

Layers 19--26 use 16 copied Qwen SwiGLU neuron groups and hard top-4 routing.
Each child keeps the parent gate/up/down slice weights, adds a zero-start
rank-8 correction, and trains for 300 local steps with only the final 50
steps aligned to hard dispatch. The benchmark uses batch 8 x 128 and grouped
dispatch.

## Results

| seed | alpha=0 CE delta | teacher top-1 | sparse/parent timing | quality gate |
|---:|---:|---:|---:|:---:|
| 2026 | +0.0413 | 92.16% | 1.101x | pass |
| 2027 | +0.0412 | 91.53% | 1.019x | pass |

Both seeds pass the `+0.05` CE-delta criterion. The active expert-body
fraction is 4/16 = 25%; stored parameters remain approximately parent-sized
because the bank contains the copied Qwen slices plus router/correction
parameters.

## Decision

This is a **synthetic transfer-control pass only**. It remains the best result
under the historical repeated-template evaluation, but V0.158 shows that the
same group path fails on held-out varied text. Do not treat this as a general
language-model quality pass or scale it to a larger model yet.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e16k4_grouped_calrank8_hard50_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e16k4_grouped_calrank8_hard50_b8s128_seed2027.json`
