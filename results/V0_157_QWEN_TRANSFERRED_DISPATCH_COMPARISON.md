# V0.157 Qwen Transferred-Neuron Dispatch Comparison

## Question

Does grouped dispatch improve runtime without changing the sparse quality
result relative to the selected-token loop?

## Results

The same 16-group/top-4, 8-layer, 25%-active seed-2026 configuration was
measured with the two dispatch implementations.

| dispatch | alpha=0 CE delta | teacher top-1 | sparse/parent timing |
|---|---:|---:|---:|
| grouped | +0.0413 | 92.16% | 1.101x |
| token-loop | +0.0789 | 92.36% | 1.031x |

The two implementations are numerically close for a fixed bank/input
(`1.68e-8` maximum absolute error in the isolated check), but tiny differences
can change top-k choices in later layers and therefore alter the sequential
calibration trajectory. In this run grouped dispatch gives the better quality
trajectory, while token-loop is faster but still slower than the dense parent.

## Decision

**Keep grouped dispatch for the quality reference and treat token-loop as a
runtime control.** A fused implementation must preserve route decisions or
include a route-stability test; speed alone is not enough if the cascade’s
quality changes.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e16k4_grouped_calrank8_hard50_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e16k4_tokenloop_calrank8_hard50_b8s128_seed2026.json`
