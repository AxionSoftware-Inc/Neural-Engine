# V0.170 — Bounded Output-Contract Audit

## Question

V0.169 showed that final-logit layerwise refinement breaks the local hidden
state contract. This audit tests a cheaper explicit contract: after each
sparse child is trained, apply a bounded per-channel affine mapping so its
output mean and standard deviation match the parent FFN output on calibration
tokens. The rank-64 cross-group correction and frozen 70-class subset router
remain unchanged.

## Protocol

The run matches the V0.167 four-layer control: Qwen3-0.6B, float32 CUDA,
layers `23--26`, 8 contiguous groups with top-4 active, rank-64 cross-group
correction, subset-router supervision for 100 steps, 300 soft child steps and
100 hard steps at `1e-4`. Calibration and evaluation use the explicit
`data/qwen_calibration.txt` and `data/qwen_eval.txt` files. The contract scale
is clipped to `[0.25, 4.0]` per hidden channel and is fitted from calibration
tokens after child training.

## Results

| Variant | Seed | Held-out +CE | Top-1 | Gate |
|---|---:|---:|---:|:---:|
| V0.167 frozen subset-router control | 2026 | `+0.0095` | 82.23% | PASS |
| Bounded output-contract + cross-group | 2026 | `+0.1549` | 78.59% | FAIL |

The contract reduces the first layer's local held-out MSE from the uncontracted
run's large drift to `13.05`, but this does not translate into end-to-end
quality. The sparse bank timing in this Python implementation was `1.33x` the
parent, so the contract is not a speed improvement either.

## Interpretation

Matching first and second moments is not enough: the next transformer block
depends on token-level directions and correlations, not only channel-wise
statistics. This result rejects output-statistics normalization as the hidden
state contract for the current circuit decomposition.

## Decision

Reject the bounded output-contract path. The best learned endpoint remains the
four-layer frozen subset-router configuration at `+0.0095` on seed 2026 and
`+0.0454` on seed 2027. Do not scale to 700M/1B yet. Further architecture
work must preserve token-level residual geometry while staying sparse; simple
loss changes, residual capacity, and marginal-statistics alignment have all
failed the matched gate.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_subsetrouter100_frozen_hard100_lr1e-4_outputcontract_seed2026.json`
