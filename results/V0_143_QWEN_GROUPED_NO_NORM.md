# V0.143 Qwen Grouped Dispatch Without Redundant Child Norm

## Question

Does removing the child LayerNorm improve the Qwen circuit-bank execution
without hurting the teacher-transfer quality? The Qwen layer supplies a
preceding RMSNorm before its MLP input, so the child LayerNorm may be redundant
in this transplant setting.

## Protocol

The V0.142 four-expert top-1 grouped bank is repeated with
`--child-no-norm`, the same Qwen3-0.6B layer 26, inner width, data, and 300-step
distillation budget. Results use 30 CUDA warmup iterations and 100 synchronized
serial timing iterations. The first 50-iteration seed-2026 measurement had a
large timing outlier, so the longer repeat is the reported timing reference.

## Results

| seed | parent FFN mean ms | normless grouped bank mean ms | bank/parent | end-to-end bank/parent | quality gate |
|---:|---:|---:|---:|---:|:---:|
| 2026 | 0.802 | 0.848 | 1.057x | 1.014x | pass |
| 2027 | 0.805 | 0.809 | 1.005x | 1.015x | pass |

The alpha=0 CE deltas are `+0.0042` and `−0.0161`, with teacher top-1
agreement `95.31%` and `95.70%`. The active expert-body fraction remains 25%.
The normless child has 3,283,076 parameters, or 34.79% of the parent FFN;
the normed child has 3,291,268 parameters, or 34.88%.

## Interpretation

Removing the redundant child norm improves the grouped implementation versus
V0.142's `1.166x` and `1.204x` isolated ratios, and it preserves the quality
gate on both seeds. The remaining small gap is now mostly dispatch/scheduling
overhead rather than the child projection arithmetic. End-to-end Qwen timing
is still about 1.5% slower, so this is not yet a demonstrated speedup.

## Decision

**Accept normless grouped dispatch as the preferred Qwen transplant path, but
keep the performance claim at neutral until a repeatable ratio below 1.0x is
measured.** Keep the generic Neural Engine block's norm option for standalone
use; the normless option is specific to a Qwen layer whose input is already
normalized. The next performance step is a fused grouped kernel or a larger
token batch where grouped matmuls can amortize routing overhead. Do not scale
to more layers or a larger model solely on this timing result.

## Artifacts

- `benchmark_qwen_two_layer_transplant.py`
- `benchmark_qwen_single_layer_bank.py`
- `results/runs/qwen_single_layer_bank4x1_grouped_nonorm_seed2026_repeat.json`
- `results/runs/qwen_single_layer_bank4x1_grouped_nonorm_seed2027_repeat.json`
