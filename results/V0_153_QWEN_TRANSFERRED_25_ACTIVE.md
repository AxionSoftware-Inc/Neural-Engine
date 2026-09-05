# V0.153 Qwen Transferred Neurons at 25% Active Compute

## Question

Can the transferred-neuron bank reduce active intermediate computation from
50% to 25% while preserving the eight-layer cascade quality gate?

## Results

With 8 neuron groups and top-2 routing across layers 19--26, seed 2026
reaches alpha=0 CE delta `+1.3736`, teacher top-1 agreement `69.21%`, and
fails the quality gate. The local error grows strongly toward the later
layers, so this is not an acceptable sparse operating point.

An ablation also tested the mathematically unbiased-looking `E` scale for a
top-k partial sum instead of the empirically stable `E/K` scale. At 8 groups
and top-4, the result is even worse: CE delta `+1.1333` and top-1 agreement
`67.92%`. Contribution-ranked selection over-selects high-energy groups, so
the unbiased random-subset correction over-amplifies the output.

## Decision

**Reject 25% active compute with the current coarse partition and scaling.**
Keep 50% active as the current quality default and retain `E/K` scaling. The
next sparse-quality experiment is finer neuron partitioning at the same 25%
active fraction, not a larger teacher or more local training steps.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k2_grouped_calrank8_hard100_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_scaledE_grouped_calrank8_hard100_b8s128_seed2026.json`
