# V0.173 — Full-Active Route-Scale Fix and Clean Depth Control

## Finding

The exactness smoke-test found a benchmark edge-case: when all `E` groups are
active, the hard route uses a softmax over `E` groups. The old default scale
`E/K=1` therefore returned the average of the group outputs instead of their
sum. Direct parent-vs-child testing showed MSE about `0.72` before the fix.
The default is now `E` when `K=E`, while the existing `E/K` rule is retained
for `K<E`.

After the fix, a full-active child reconstructs its parent at float32 noise
(`max abs 1.9e-6`, MSE `1.2e-13`). The same rule is applied to the individual
neuron child.

## Clean depth control

The corrected variable-depth run uses layers `19--26`, active schedule
`8,8,8,8,4,4,4,4`, correction-rank schedule `0,0,0,0,64,64,64,64`, and
hard-scale schedule `8,8,8,8,4,4,4,4`. The first four layers are now exact
transfers; only the last four use the rank-64 sparse correction.

| Variant | Seed | Held-out +CE | Top-1 | Gate |
|---|---:|---:|---:|:---:|
| Corrected full-active prefix + late sparse suffix | 2026 | `+0.7671` | 68.68% | FAIL |

The clean control removes the earlier full-active scaling artifact, but still
fails substantially. Thus the depth cascade remains a real problem even when
the early prefix is mathematically exact; it is not explained by the earlier
K=E scale bug.

## Decision

Keep the scale fix in the benchmark and invalidate no prior `K=4` results,
because their default `E/K=2` is unchanged. Reject the 8-layer late-suffix
configuration at this training/circuit setting and do not scale to 700M/1B.
The four-layer frozen subset-router result remains the best learned endpoint.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_crossgroup_8layers_e8_schedule8844_rank0x4_rank64x4_scale8444_subsetrouter100_seed2026.json`
