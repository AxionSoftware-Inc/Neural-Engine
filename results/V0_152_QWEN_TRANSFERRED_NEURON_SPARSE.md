# V0.152 Qwen Transferred-Neuron Sparse Cascade

## Question

Can copied Qwen intermediate neurons be partitioned into a sparse routed bank
while preserving the quality of the full parent function across multiple
layers?

## Protocol

Each Qwen SwiGLU intermediate dimension is partitioned contiguously across
experts. The expert slices keep the parent `gate/up/down` weights; only the
router and a zero-start rank-8 hidden correction are trainable. Hard routing
selects a fixed fraction of the neuron groups per token. The local training
uses 300 steps with a final 100-step hard-route phase.

## Results

| layers | experts/top-k | seed | alpha=0 CE delta | teacher top-1 | sparse/parent timing | gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 25--26 | 4/2 | 2026 | +0.0323 | 96.85% | 1.001x | pass |
| 23--26 | 4/2 | 2026 | +0.0048 | 97.39% | 0.992x | pass |
| 19--26 | 4/2 | 2026 | +0.0602 | 92.94% | 0.997x | fail |
| 19--26 | 4/2 | 2027 | -0.0041 | 93.65% | 0.994x | pass |
| 19--26 | 8/4 | 2026 | +0.0129 | 93.21% | 1.029x* | pass |
| 19--26 | 8/4 | 2027 | -0.0081 | 93.85% | 1.027x* | pass |

The 8-expert/top-4 design keeps active expert-body compute at 50% while
making the neuron groups half as large. It removes the 4-expert seed failure
and passes the eight-layer quality gate on both seeds. The asterisk marks the
initial token-loop timing; after implementing the actual grouped dispatcher,
the seed-2026 timing was `1.054x`, so no speedup claim is made yet.

The sparse bank stores essentially the full parent intermediate weights plus
router/correction parameters (about 1.016x parent FFN parameters per child),
so this result is an active-compute result, not a memory-compression result.

## Decision

**Accept 8-expert/top-4 transferred-neuron routing as the current quality
default for continued work.** It is the strongest multi-layer sparse result
so far and directly uses Qwen weights without re-learning the FFN. Keep the
performance gate open: the next engineering task is a fused/low-overhead
grouped implementation, followed by a 25% active-compute test.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_2layers_calrank8_hard100_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_4layers_calrank8_hard100_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_calrank8_hard100_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_calrank8_hard100_b8s128_seed2027.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_calrank8_hard100_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_calrank8_hard100_b8s128_seed2027.json`
- `results/runs/qwen_multi_layer_transferred_neuron_sparse_8layers_e8k4_grouped_calrank8_hard100_b8s128_seed2026.json`
