# V0.167 — Coupled Subset Router and Depth-Cascade Audit

## Question

The exact best-subset control showed that dot/energy scores were not a valid
proxy for the group combination needed by the teacher. This audit replaces
independent group scores with a coupled classifier over all `C(8,4)=70`
possible subsets, freezes that router after supervision, and then tests the
hard-route schedule at two, four, and eight layers.

## Method

For each calibration token, every group output is evaluated only during
router supervision. The target is the 4-group subset minimizing the exact
teacher-output MSE under the benchmark's hard scaling. A 70-class hidden-state
router predicts that subset. During child distillation the subset router is
frozen, preventing the soft reconstruction objective from erasing the
supervised routing function. In the hard path, the selected subset uses the
existing grouped sparse dispatch.

Unless noted, runs use Qwen3-0.6B, float32 CUDA, 8 groups/top-4 active,
contiguous Qwen FFN slices, cross-group rank-64 correction, 300 child steps,
and 200 hard-route steps at `3e-4`. The gentler 4-layer schedule uses 100 hard
steps at `1e-4`.

## Results

| Variant | Layers | Seed | Held-out +CE | Top-1 | Gate |
|---|---:|---:|---:|---:|:---:|
| Exact best-subset oracle, rank0 | 25--26 | 2026 | `+0.0384` | 90.48% | PASS* |
| Exact best-subset oracle, rank0 | 23--26 | 2026 | `+0.0439` | 89.16% | PASS* |
| Coupled subset router, unfrozen | 25--26 | 2026 | `+0.0427` | 85.13% | PASS |
| Coupled subset router, unfrozen | 25--26 | 2027 | `+0.0745` | 84.69% | FAIL |
| Coupled subset router, frozen | 25--26 | 2026 | `+0.0109` | 86.72% | PASS |
| Coupled subset router, frozen | 25--26 | 2027 | `+0.0415` | 86.50% | PASS |
| Coupled subset router, frozen, hard100/lr1e-4 | 23--26 | 2026 | `+0.0095` | 82.23% | PASS |
| Coupled subset router, frozen, hard100/lr1e-4 | 23--26 | 2027 | `+0.0454` | 81.13% | PASS |
| Coupled subset router, frozen, hard100/lr1e-4 | 19--26 | 2026 | `+1.4731` | 47.66% | FAIL |
| Coupled subset router, frozen, hard100/lr1e-5 | 19--26 | 2026 | `+0.5685` | 60.47% | FAIL |

`*` The oracle runs are diagnostic upper-bound-style controls: all groups are
computed to choose the subset, so their timing is not a sparse deployment
claim. Their timing iterations were shortened for practicality.

## Interpretation

The exact oracle passes at both two and four layers. This overturns the earlier
overly strong interpretation of the dot-oracle failure: the contiguous copied
cells do contain subsets that preserve the teacher function under the current
50%-active output rule. The primary bottleneck is predicting those subsets
cheaply and robustly, not proving that every sparse subset is inadequate.

The coupled subset router becomes reliable at two layers when its supervised
weights are frozen before child distillation. At four layers, the same freeze
plus a gentler hard phase passes both seeds. It improves the prior seed-2026
cross-group reference (`+0.0181` versus `+0.0095`), while seed 2027 is within
the gate but weaker than its prior reference (`+0.0155` versus `+0.0454`). The
4-layer result is a valid research endpoint, not yet a production speed result:
current grouped Python timing is
about `1.32x` parent and copied buffers remain about `2.14x` parent storage.

At eight layers, lowering the hard learning rate prevents the single-layer
explosion seen at `1e-4`, but the cascade still fails (`+0.5685`). Errors grow
as earlier sparse replacements change later-layer hidden states. This is a
depth-composition problem requiring layer-aware joint training or a stable
residual interface; it is not fixed by simply increasing model capacity.

## Decision

Freeze the current 4-layer method as the best learned sparse checkpoint
configuration for further research:

```text
Qwen3-0.6B, layers 23--26
8 groups / top-4 active
contiguous copied FFN cells
rank-64 cross-group correction
70-class best-subset router
router frozen after 100 supervision steps
100 hard steps at lr=1e-4
```

Do not move to 700M/1B or claim full-stack attention-free quality. The next
high-value experiment is a layer-aware cascade interface: train each router
against the hidden-state distribution produced by the already sparse prefix,
then jointly calibrate only the failing depth boundary while keeping the
validated 4-layer result as a frozen control.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_crossgroup_2layers_e8k4_rank0_oraclesubset_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank0_oraclesubset_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_2layers_e8k4_rank64_subsetrouter100_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_2layers_e8k4_rank64_subsetrouter100_seed2027.json`
- `results/runs/qwen_multi_layer_crossgroup_2layers_e8k4_rank64_subsetrouter100_frozen_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_2layers_e8k4_rank64_subsetrouter100_frozen_seed2027.json`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_subsetrouter100_frozen_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_subsetrouter100_frozen_seed2027.json`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_subsetrouter100_frozen_hard100_lr1e-4_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_subsetrouter100_frozen_hard100_lr1e-4_seed2027.json`
- `results/runs/qwen_multi_layer_crossgroup_8layers_e8k4_rank64_subsetrouter100_frozen_hard100_lr1e-4_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_8layers_e8k4_rank64_subsetrouter100_frozen_hard100_lr1e-5_seed2026.json`
