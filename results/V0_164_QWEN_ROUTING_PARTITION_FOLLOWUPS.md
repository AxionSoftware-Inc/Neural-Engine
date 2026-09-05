# V0.164 — Qwen routing, partition, and teacher-initialization follow-ups

## Scope

This audit continues the held-out Qwen3-0.6B FFN-transfer protocol on
`data/qwen_eval.txt`: float32 CUDA, 300 local child steps, 200 hard-route
steps at learning rate `3e-4`, and the same teacher-forced CE gate of `+0.05`.
The validated reference is contiguous 8-group/top-4 cross-group mixing on
four late layers, which reached `+0.0181` and `+0.0155` on two seeds.

## Results

| Variant | Layers / active body | Held-out +CE | Top-1 | Gate |
|---|---:|---:|---:|:---:|
| Teacher full-output decoder, rank 64 | 2 / 50% | `+0.1455` | 83.50% | FAIL |
| Teacher residual initializer, rank 64, short smoke | 2 / 50% | `+0.6216` | 63.31% | FAIL |
| Activation-balanced partition + cross-group | 2 / 50% | `+0.0456` | 84.40% | PASS |
| Activation-balanced partition + cross-group | 4 / 50% | `+0.0724` | 81.81% | FAIL |
| Random sampled-overlap partition | 4 / 50% | `+0.0079` | 84.40% | PASS |
| Random sampled-overlap partition, second seed | 4 / 50% | `+0.0983` | 78.78% | FAIL |
| Random sampled-overlap, hard scale 8 | 4 / 50% | `+0.2620` | 67.33% | FAIL |
| Stratified-overlap partition | 4 / 50% | `+0.0673` | 78.81% | FAIL |
| E=16, K=8 cross-group | 4 / 50% | `+0.0330` | 80.20% | PASS |
| Group router energy supervision | 4 / 50% | `+0.0312` | 80.66% | PASS |
| Group router teacher-dot supervision | 8 / 50% | `+0.1834` | 62.18% | FAIL |
| Sensitivity-selected five sparse layers | 5 / 31.25% global saving | `+0.0292` | 74.32% | PASS |
| Sensitivity schedule, K=6 only on selected layers | 8 / 59.4% active | `+0.0975` | 66.48% | FAIL |
| Individual-neuron router, 50% active | 2 / 50% | `+0.3497` | 71.09% | FAIL |
| Individual-neuron oracle-dot, 50% active | 2 / 50% | `+0.4032` | 70.70% | FAIL |
| Individual-neuron oracle-energy, 50% active | 2 / 50% | `+0.4292` | 68.75% | FAIL |

The random overlap pass is not reproducible across seeds and is rejected.
Activation balancing, finer E=16/K=8 routing, router supervision, and
teacher-derived residual initialization do not beat the contiguous
cross-group reference. Increasing K only on sensitive layers also does not
repair the eight-layer cascade.

The individual-neuron diagnostics are worse even with an oracle route. This
means the current top-k neuron approximation and fixed `I/K` rescale lose
the nonlinear sum before router prediction becomes the only problem. A fused
kernel would improve speed, not this quality failure.

## Teacher-derived decoder decision

Fitting a low-rank map from one copied group output to the full teacher FFN
output is the wrong target at rank 64: it asks one small nonlinear slice to
reconstruct all omitted nonlinear terms. Rank384 was attempted, but its
1024×1024 factorizations and wide decoder training were impractical on the
available 12 GB GPU; it produced no result file before being stopped. The
teacher-residual variant was also rejected because its hard-route residual
target is not stable across selected group combinations.

## Decision

Keep the contiguous cross-group four-layer result as the current conditional
proof. The five-layer sensitivity subset is a useful partial-sparsity
operating point, but it is not a full Transformer-free or all-FFN solution.
Do not scale the new partition or teacher-decoder variants to 300M--1B.

The remaining high-value architecture work is a fused individual-neuron or
functionally coherent micro-group implementation with a selection objective
that preserves the sum of nonlinear neuron contributions. The current Python
group bank is a research diagnostic, not yet a production sparse kernel.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_transferred_teacher_group_decoder_2layers_e8k4_rank64_seed2026.json`
- `results/runs/qwen_multi_layer_teacher_group_residual_2layers_e8k4_rank64_hard50_router100_seed2026.json`
- `results/runs/qwen_multi_layer_activation_balanced_crossgroup_2layers_e8k4_rank64_seed2026.json`
- `results/runs/qwen_multi_layer_activation_balanced_crossgroup_4layers_e8k4_rank64_seed2026.json`
- `results/runs/qwen_multi_layer_sampled_overlap_crossgroup_4layers_e8k4_rank64_seed2026.json`
- `results/runs/qwen_multi_layer_sampled_overlap_crossgroup_4layers_e8k4_rank64_seed2027.json`
- `results/runs/qwen_multi_layer_sampled_overlap_crossgroup_4layers_e8k4_rank64_scale8_seed2026.json`
- `results/runs/qwen_multi_layer_stratified_overlap_crossgroup_4layers_e8k4_rank64_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e16k8_rank64_scale4_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_router100_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_8layers_e8k4_routerdot100_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_sensitivity5_e8k4_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_sensitivity_schedule_e8_seed2026.json`
- `results/runs/qwen_neuron_sparse_2layers_active1536_seed2026.json`
- `results/runs/qwen_neuron_sparse_2layers_active1536_oracle_dot_seed2026.json`
- `results/runs/qwen_neuron_sparse_2layers_active1536_oracle_energy_seed2026.json`
