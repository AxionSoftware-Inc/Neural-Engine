# V0.150 Qwen Eight-Layer Cascade Failure Analysis

## Question

Why does the rank-8 random sparse child pass at four layers but fail when the
replacement block is expanded to eight layers?

## Results

All runs use Qwen3-0.6B, layers 19--26, four experts with top-2 grouped
routing, no child norm, 384-wide children, batch 8 x 128, and 300 local
distillation steps.

| variant | seed | alpha=0 CE delta | teacher top-1 | gate |
|---|---:|---:|---:|:---:|
| soft local training | 2026 | +0.2192 | 87.77% | fail |
| soft local training | 2027 | +0.3714 | 86.38% | fail |
| final 100 steps hard-route aligned | 2026 | +0.2055 | 89.36% | fail |
| hard-route aligned + 50-step joint logit refine | 2026 | +0.1935 | 89.06% | fail |
| hard-route aligned, inner 768 | 2026 | +0.2533 | 88.55% | fail |

The useful control is alpha=0.75: it passes on the failing baseline runs,
showing that each child function is locally useful but the full handoff
accumulates representation error. Hard-route-aligned local training lowers
the local MSE and improves the final result slightly, while joint logit
refinement gives only a small additional reduction. Doubling child width
does not repair the cascade and makes the full handoff worse in this budget.

## Decision

**Reject raw random-child scaling as the eight-layer architecture path.** The
failure is not explained by insufficient nominal width alone. Preserve the
four-layer rank-8 result, and switch to direct Qwen SwiGLU weight transfer so
the circuit begins from the parent function instead of learning it from
scratch.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_routed_grouped_nonorm_calrank8_8layers_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_routed_grouped_nonorm_calrank8_8layers_b8s128_seed2027.json`
- `results/runs/qwen_multi_layer_routed_grouped_nonorm_calrank8_8layers_hard100_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_routed_grouped_nonorm_calrank8_8layers_hard100_joint50_b8s128_seed2026.json`
- `results/runs/qwen_multi_layer_routed_grouped_nonorm_calrank8_8layers_inner768_hard100_b8s128_seed2026.json`
