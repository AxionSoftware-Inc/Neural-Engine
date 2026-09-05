# V0.149 Qwen Four-Layer Rank-8 Cascade

## Question

Does the rank-8, normless, grouped sparse child remain stable when the
replacement is expanded from two adjacent Qwen MLPs to four layers, and does
the full model show a repeatable timing benefit?

## Protocol

Layers 23--26 of the cached Qwen3-0.6B model were replaced sequentially. Each
child was trained locally for 300 steps after the preceding child handoff, so
the later children saw the representation produced by the earlier sparse
children. Every child used four experts with hard top-2 routing, grouped
dispatch, no internal child norm, and a zero-start rank-8 calibration
correction. The batch was 8 x 128 tokens. Parent and four-layer sparse
versions were measured with 20 CUDA warmup iterations and 50 synchronized
serial iterations over the same evaluation batches.

## Results

| seed | alpha=0 CE delta | teacher top-1 | parent mean ms | four-layer sparse mean ms | sparse/parent | quality gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 2026 | +0.0298 | 94.29% | 224.802 | 220.784 | 0.982x | pass |
| 2027 | +0.0376 | 94.24% | 228.817 | 222.155 | 0.971x | pass |

Both seeds pass the quality gate. Each child has 3,299,460 scalar parameters,
or 34.96% of its parent FFN, while the hard route executes two of four expert
bodies per token (50% expert-body activity).

## Interpretation

This is the first repeatable positive full-model timing signal for a
multi-layer sparse cascade in this benchmark: both 50-iteration runs are
faster, by about 1.8% and 2.9%, while retaining stable teacher agreement.
The gain is intentionally modest because only four of Qwen's 28 layers are
replaced and the Python grouped dispatcher still has runtime overhead. The
result supports scaling the same architecture to a larger adjacent block;
it does not yet justify claiming a general speedup or moving to a larger
teacher model.

## Decision

**Accept the four-layer rank-8 cascade as a quality and engineering GO.** The
next gate is an eight-layer adjacent cascade with the same configuration and
two seeds. Keep the 0.6B teacher until that larger replacement block is
validated; this separates architecture scaling from teacher-size effects.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_routed_grouped_nonorm_calrank8_timing50_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_routed_grouped_nonorm_calrank8_timing50_b8s128_seed2027.json`
