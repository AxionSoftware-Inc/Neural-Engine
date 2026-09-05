# V0.159 Qwen Individual-Neuron Sparse Smoke

## Question

Can individual copied Qwen SwiGLU neurons replace coarse groups while
activating only a selected fraction of the parent intermediate dimension?

## Results

A first 2-layer smoke on layers 25--26 used 768 of 3072 neurons per token
(25%), rank-8 correction, 10 total steps, and 2 hard-route steps. On the
held-out corpus it reached `+0.3487` CE delta and 71.09% teacher top-1
agreement. The Python gather path was `2.75x` the dense parent even on this
small smoke.

The gather implementation was then changed to process tokens in chunks of 64,
which removes the large temporary `[tokens, active_neurons, hidden]` tensor.
With the same 2-layer setup but 100 total steps and 20 hard-route steps, the
chunked run reached `+0.2949` held-out CE delta and 75.20% teacher top-1
agreement. It still took `2.97x` the dense parent. Chunking fixes the memory
failure mode, but not the quality or runtime problem.

An 8-layer 1536-neuron run was stopped before completion because the original
prototype materialized a token-by-active-neuron-by-hidden temporary tensor and
became impractical. Chunking addresses that specific allocation, but the
2-layer result shows that the current Python gather path is still not a viable
scale-up route. This remains an implementation limitation, not a final
mathematical rejection of individual-neuron routing.

## Decision

**Do not train the full neuron-level model with the current Python gather
code.** The next implementation must use a fused selected-neuron matmul (or a
compiled equivalent) and must report copied-buffer storage separately from
trainable router parameters. The accounting and chunked-memory fixes are now
in the benchmark.

## Artifact

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_neuron_sparse_2layers_smoke_active768_seed2026.json`
- `results/runs/qwen_neuron_sparse_2layers_active768_readme_eval_steps100_seed2026.json`
