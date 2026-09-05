# V0.169 — Layerwise Task-Loss Refinement Audit

## Question

V0.168 showed that generic joint refinement destabilizes the eight-layer
cascade. This audit tests a more constrained alternative: refine one sparse
child at a time against final teacher logits, with earlier layers already
sparse and later layers still dense parents. The goal is to give each layer a
task-aware gradient without letting a joint optimizer trade errors across all
interfaces.

## Protocol

The run matches the V0.167 four-layer control: Qwen3-0.6B, float32 CUDA,
layers `23--26`, 8 contiguous groups with top-4 active, rank-64 cross-group
correction, 100-step subset-router supervision, 300 soft child steps and 100
hard steps at `1e-4`. It uses the explicit calibration/evaluation files
`data/qwen_calibration.txt` and `data/qwen_eval.txt`. After each child is
trained, it receives 20 per-layer final-logit refinement steps at learning rate
`1e-5`; the active layer is sparse, the prefix is sparse, and the suffix is
dense during that refinement.

## Results

| Variant | Seed | Held-out +CE | Top-1 | Gate |
|---|---:|---:|---:|:---:|
| V0.167 frozen subset-router control | 2026 | `+0.0095` | 82.23% | PASS |
| Layerwise task-loss refinement, 20 steps/layer | 2026 | `+0.4703` | 69.43% | FAIL |

The first layer's local held-out MSE also rose to `135.04` after refinement,
and the per-layer task loss remained high (`42.62`, `63.99`, `69.43`, `83.32`)
at the final recorded step.

## Interpretation

The constrained final-logit objective is not a safe replacement for local FFN
distillation. It finds parameter changes that can reduce the selected
task-level loss during the layerwise pass but destroy the local function that
the next sparse layer depends on. The failure at four layers is decisive; no
eight-layer follow-up is justified.

This also strengthens the diagnosis: the remaining issue is not merely the
choice between local MSE and a final-logit objective. The sparse interface must
preserve a stable hidden-state contract while approximating the omitted FFN
cells.

## Decision

Reject layerwise task-loss refinement. Keep local sparse-child distillation and
the frozen subset router as the validated four-layer research endpoint. Do not
scale to 700M/1B yet. Any next architecture should introduce an explicit,
bounded hidden-state contract or error-correction channel and must first beat
the `+0.0095` four-layer control at matched protocol.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_crossgroup_4layers_e8k4_rank64_subsetrouter100_frozen_hard100_lr1e-4_layerwise20_lr1e-5_seed2026.json`
