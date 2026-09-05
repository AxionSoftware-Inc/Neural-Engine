# V0.148 Qwen Two-Layer Rank-8 Timing

## Question

Does the rank-8 interface correction preserve the sparse execution advantage
when two adjacent Qwen MLPs are replaced at once?

## Protocol

The V0.147 normless four-expert top-2 grouped bank with rank-8 zero-start
calibration is trained independently for layers 25 and 26. The model uses
batch size 8 and sequence length 128. Parent and sparse versions are measured
for 20 CUDA warmup and 30 synchronized serial iterations over the same four
evaluation batches. The timing is full-model Qwen forward with both adjacent
MLPs replaced; no joint refinement is used.

## Results

| seed | parent mean ms | two-layer sparse mean ms | bank/parent | quality gate |
|---:|---:|---:|---:|:---:|
| 2026 | 234.095 | 225.530 | 0.963x | pass |
| 2027 | 225.613 | 235.453 | 1.044x | pass |

The corresponding quality deltas are `+0.0184` and `+0.0289`, with teacher
top-1 agreement `96.53%` and `96.66%`. Rank-8 correction remains only 16,384
parameters per child, and the hard route executes two of four expert bodies
per token.

## Interpretation

Quality is stable across seeds, but end-to-end timing is not: one seed is 3.7%
faster and the other 4.4% slower. The combined mean is effectively neutral,
so the result does not support a two-layer speedup claim. This is compatible
with V0.144's strong isolated FFN speedup: the unchanged Qwen layers and
launch/runtime variation dominate a small two-layer full-forward delta.

## Decision

**Keep rank-8 calibration as the quality path, but classify two-layer
end-to-end performance as neutral.** Do not move to 700M/1B based on this
timing alone. The next engineering gate is repeated timing with a more stable
measurement method and, if necessary, a fused grouped kernel; the architecture
must retain both the rank-8 quality pass and a repeatable performance result.

## Artifacts

- `benchmark_qwen_two_layer_transplant.py`
- `results/runs/qwen_two_layer_routed_grouped_nonorm_calrank8_timing_b8s128_seed2026.json`
- `results/runs/qwen_two_layer_routed_grouped_nonorm_calrank8_timing_b8s128_seed2027.json`
