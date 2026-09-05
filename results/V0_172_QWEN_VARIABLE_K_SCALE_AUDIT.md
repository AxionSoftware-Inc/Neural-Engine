# V0.172 — Variable-K Route-Scale Audit

## Question

V0.171 showed that a depth-dependent active schedule `6,6,6,6,4,4,4,4`
fails when each layer uses the default `E/K` hard scale. This audit keeps the
same schedule but removes that scale discontinuity: 6-active layers use scale
`1.0`, while 4-active layers retain scale `4.0`.

## Results

| Variant | Seed | Held-out +CE | Top-1 | Gate |
|---|---:|---:|---:|:---:|
| V0.171 schedule, default `E/K` scale | 2026 | `+5.8722` | 6.35% | FAIL |
| Same schedule, early scale `1.0` | 2026 | `+4.5923` | 17.90% | FAIL |

The scale change improves the failure numerically but remains far outside the
`+0.05` gate. Multiple early and late child losses still spike, and timing is
about `1.89x` the dense parent in the current Python implementation.

## Interpretation and decision

Route-scale discontinuity is part of the instability, but not the root cause.
The variable-K schedule remains unusable even after this correction. Reject
variable active schedules and manual scale tuning as the current path; keep the
validated four-layer frozen subset-router result as the control. The next
architecture must change how sparse cells compose across depth, rather than
only changing `K` or its multiplier.

## Artifact

- `results/runs/qwen_multi_layer_crossgroup_8layers_e8_schedule6644_scale1144_rank64_subsetrouter100_seed2026.json`
