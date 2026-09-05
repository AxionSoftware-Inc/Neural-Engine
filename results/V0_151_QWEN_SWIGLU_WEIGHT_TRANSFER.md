# V0.151 Qwen SwiGLU Weight Transfer

## Question

Can a Qwen3 FFN be converted into an attention-free Neural Engine circuit
without retraining by copying its `gate_proj`, `up_proj`, and `down_proj`
weights?

## Protocol

The three bias-free Qwen3 SwiGLU projections were copied into the equivalent
attention-free `SiLU(gate(x)) * up(x) -> down(x)` child for layers 19--26.
No child training and no calibration correction were used. The eight-layer
replacement was evaluated on the same batch 8 x 128 gate as V0.150.

## Results

Each transferred child has 9,444,352 parameters versus 9,437,184 in the
parent FFN; the small difference is from the child’s explicit zero biases.
Local function MSE is `2.0e-12` to `6.1e-12`. At alpha=0, CE delta is `0.0`
and teacher top-1 agreement is `99.95%`. This is exact functional transfer
up to floating-point roundoff, not learned imitation.

Full-model timing is `236.494 ms` versus `227.330 ms` for the parent, or
`1.040x`, so the dense conversion alone is not a speedup and does not reduce
stored parameter count.

## Decision

**Accept weight transfer as the initialization/compilation path, not as the
final sparse design.** It proves that Qwen’s trained FFN can seed Neural
Engine circuits without hours of retraining. The next step is to partition
the copied intermediate neurons into routed groups and measure the quality
versus active compute.

## Artifact

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_qwen_transfer_8layers_b8s128_seed2026.json`
