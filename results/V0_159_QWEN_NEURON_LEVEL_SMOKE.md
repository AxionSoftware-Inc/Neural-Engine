# V0.159 Qwen Individual-Neuron Sparse Smoke

## Question

Can individual copied Qwen SwiGLU neurons replace coarse groups while
activating only a selected fraction of the parent intermediate dimension?

## Results

A 2-layer smoke on layers 25--26 used 768 of 3072 neurons per token (25%),
rank-8 correction, 10 total steps, and 2 hard-route steps. On the held-out
corpus it reached `+0.3487` CE delta and 71.09% teacher top-1 agreement. The
Python gather path was `2.75x` the dense parent even on this small smoke.

An 8-layer 1536-neuron run was stopped before completion because the prototype
materialized a token-by-active-neuron-by-hidden temporary tensor and became
impractical. This is a runtime limitation of the implementation, not a final
mathematical rejection of individual-neuron routing.

## Decision

**Do not train the full neuron-level model with the current gather code.** The
next implementation must use chunked selected-neuron matmuls or a fused CUDA
kernel, and it must report copied-buffer storage separately from trainable
router parameters. The accounting fix is now in the benchmark.

## Artifact

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_neuron_sparse_2layers_smoke_active768_seed2026.json`
