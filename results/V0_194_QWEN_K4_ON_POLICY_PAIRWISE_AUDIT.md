# V0.194 — Eight-layer K=4 on-policy pairwise router audit

**Date:** 2026-09-07  
**Status:** `REJECTED`  
**Branch:** `exp/test-p003-progressive-capacity`

## Question

Can the K=4 learned-router gap be repaired by replacing the 70-subset
classifier with a 36-component pairwise cost router and refitting it on
current-cascade (on-policy) inputs? The exact local subset oracle is good, so
this isolates router generalization rather than adding model capacity.

## Protocol

- Qwen/Qwen3-0.6B, layers 19–26;
- E=8, K=4 active, contiguous groups, 50% active expert fraction;
- rank-64 cross-group correction, `hard_route_scale=4`;
- `route_source=pairwise-cost-router`, component parameterization;
- pairwise normalized regret target, 100 initial router steps;
- 300 child distillation + 300 hard-training steps at `3e-4`;
- aggregate cascade refit: 3 rounds × 100 steps at `1e-4`;
- batch 8 × sequence 128, 8 training batches, 4 held-out batches;
- float32 CUDA, current optimized grouped correction dispatch.

The only intended quality change relative to the existing K=4 control is the
pairwise router plus cascade aggregation. The exact JSON artifacts are kept in
the ignored local `results/runs/` directory:

- `qwen_v0194_8layers_k4_pairwise_aggregate_optcorr_seed2026.json`
- `qwen_v0194_8layers_k4_pairwise_aggregate_optcorr_seed2027.json`

## Result

| seed | learned CE delta | mean local regret | mean layer p95 regret | timing / parent |
|---:|---:|---:|---:|---:|
| 2026 | `+0.068222` | `0.073924` | `0.277625` | `1.129x` |
| 2027 | `+0.077447` | `0.072244` | `0.277354` | `1.125x` |

The existing eight-layer K=4 direct-hard subset-router control was
`+0.06462/+0.06165`; the new route is not better on either seed and is worse
on seed 2027. Both fail the `+0.05` quality gate. The optimized dispatch makes
the failed K=4 route fast, but speed does not repair its route quality.

## Decision

Reject pairwise cost plus on-policy aggregation for the current eight-layer
K=4 recipe. This closes another plausible router-only explanation: a richer
pair interaction head and current-cascade refit are insufficient. The local
oracle/learned split remains, so K=4 still requires a different route signal or
training objective. Do not reduce the promoted K=5 operating point to K=4
based on this result.

The next quality work should target the selector's causal objective or a
sparse output-aware signal, while keeping the current K=5/K=6 controls and
V0.193 dispatch implementation fixed.
