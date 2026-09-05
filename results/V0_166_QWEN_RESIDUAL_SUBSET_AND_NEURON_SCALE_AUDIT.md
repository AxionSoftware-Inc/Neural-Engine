# V0.166 — Residual Coreset, Best-Subset Router, and Neuron-Scale Audit

## Question

After the V0.165 oracle result, this audit tests three focused repairs before
changing the circuit decomposition: an always-on residual coreset with signed
selected-group coefficients, a router trained against the exact best group
subset under held-out teacher MSE, and removal of the individual-neuron
router's fixed `N/K` rescale.

## Protocol

Unless noted, runs use Qwen3-0.6B, float32 CUDA, held-out
`data/qwen_eval.txt`, contiguous copied Qwen groups, 8 groups/top-4 active,
300 child steps, and 200 hard-route steps. The quality gate is fully-active
sparse-vs-teacher held-out `+CE <= +0.05`.

## Results

| Variant | Layers | Seed | Held-out +CE | Top-1 | Gate |
|---|---:|---:|---:|---:|:---:|
| Residual coreset, signed coefficients + rank-64 residual | 25--26 | 2026 | `+0.1378` | 83.13% | FAIL |
| Best-subset router supervision + cross-group rank64 | 25--26 | 2026 | `+0.0407` | 85.64% | PASS |
| Best-subset router supervision + cross-group rank64 | 25--26 | 2027 | `+0.0567` | 85.47% | FAIL |
| Individual-neuron oracle-energy, fixed `N/K=2` | 25--26 | 2026 | `+0.4292` | 68.75% | FAIL |
| Individual-neuron oracle-energy, no rescale (`scale=1`) | 25--26 | 2026 | `+0.2779` | 79.17% | FAIL |

## Interpretation

The residual coreset does not repair the hard sparse path even at the easier
two-layer endpoint. The signed coefficient head and always-on rank-64
nonlinear residual add trainable capacity, but the held-out result is worse
than the existing contiguous cross-group reference (`+0.0306`). This variant
is rejected as the primary architecture.

Best-subset supervision is a more faithful router target than energy or dot
importance: one seed passes and reaches `85.64%` top-1 agreement. It is not
reproducible on the second seed (`+0.0567`), so it is not a scaling signal.
The extra target computation also makes calibration materially slower.

Removing the individual-neuron `N/K` rescale improves the oracle-energy result
by `0.1513 CE`, confirming that fixed unbiased scaling over-amplifies the
largest selected contributions. However, `+0.2779` still fails badly, so
rescale is not the fundamental fix.

## Decision

Reject residual coreset and best-subset supervision as current production
paths, and keep neuron scale=1 only as a diagnostic observation. The exact
best-subset follow-up below shows that the current copied cells do have useful
headroom; the remaining challenge is learning that subset cheaply and robustly.

The next diagnostic is an exact best-subset oracle at group level, with no
learned router and no correction wrapper. It will establish the upper bound of
the current partition under the same fixed hard output rule. That follow-up is
recorded in V0.167 and shows that the current partition has usable headroom;
the main remaining issue is learned route generalization and depth stability.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_residual_coreset_2layers_e8k4_rank64_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_2layers_e8k4_rank64_routersubset100_seed2026.json`
- `results/runs/qwen_multi_layer_crossgroup_2layers_e8k4_rank64_routersubset100_seed2027.json`
- `results/runs/qwen_neuron_sparse_2layers_active1536_oracle_energy_seed2026.json`
- `results/runs/qwen_neuron_sparse_2layers_active1536_oracle_energy_scale1_seed2026.json`
