# V0.171 — Depth-Dependent Active Schedule Audit

## Question

The eight-layer 50%-active cascade fails even with the coupled subset router.
This audit tests whether preserving more FFN capacity in the early, more
sensitive layers stabilizes the cascade without forcing every layer to use the
same active count.

## Protocol

The run uses Qwen3-0.6B, float32 CUDA, layers `19--26`, 8 contiguous groups,
rank-64 cross-group correction, the frozen 70-class subset router, explicit
`data/qwen_calibration.txt` and `data/qwen_eval.txt`, 300 soft steps and 100
hard steps at `1e-5`. The active schedule is `6,6,6,6,4,4,4,4`: the first
four layers retain 75% of groups and the last four retain 50%.

## Results

| Variant | Seed | Held-out +CE | Top-1 | Gate |
|---|---:|---:|---:|:---:|
| Depth schedule `6,6,6,6,4,4,4,4` | 2026 | `+5.8722` | 6.35% | FAIL |

The first layer's hard-training loss rose to `226.75`, and its local held-out
MSE reached `268.55`. The sparse bank was `1.88x` the parent timing in this
Python implementation; the schedule therefore did not produce a useful
quality or speed tradeoff.

## Interpretation

More active groups in the early layers are not sufficient. In this
configuration, changing `K` also changes the hard-route scale, subset-class
geometry and correction optimization, and the first layer becomes unstable.
The failure is much larger than the `+0.5685` matched eight-layer
50%-active control, so this is not evidence that the model simply needs a
larger active budget.

## Decision

Reject this depth schedule as configured and do not run a second seed. The
current four-layer frozen subset-router endpoint remains the only reliable
learned sparse result. A future variable-capacity design must use a
route-scale/training rule explicitly invariant to `K`; changing only the
active count is not enough.

## Artifact

- `results/runs/qwen_multi_layer_crossgroup_8layers_e8_schedule6644_rank64_subsetrouter100_frozen_hard100_lr1e-5_seed2026.json`
