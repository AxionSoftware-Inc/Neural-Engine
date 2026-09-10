# P-003 native dynamic-width dispatch audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Base checkpoint: 500M stable-prefix K=16, ordered factor slots, shared route
keys, rank-8 step adapter, 25% low/high edge mix

## Question

Can the K=16 quality ceiling be used selectively, so easy examples run with
K=8 while uncertain examples receive K=16? This is the first runtime-aware
prototype; it is not yet a learned width predictor.

## Implementation

The router still produces the top 16 candidates. At inference, the normalized
entropy of the top-8 route weights is used as a confidence proxy:

- normalized prefix entropy below `0.995`: execute only the first 8 circuits;
- entropy at least `0.995`: execute all 16 circuits.

The first 8 weights are renormalized before dispatch. Training remains fixed
width, because this hard choice has no width loss yet. The model now reports
both `selected_ids` (router choice) and `executed_selected_ids` (actual circuit
work), preventing the runtime audit from counting skipped slots as executed.

## Protocol

- Same two K=16 checkpoints, seeds 17/18, no additional training;
- 24 batches per condition;
- uniform, combination-heldout, low-edge `[0,7]`, and high-edge `[56,63]`;
- K=16 full execution is the quality ceiling control;
- the comparison is inference-only and compute is not claimed from wall-clock
  timing alone.

## Result: K=16 versus entropy-gated K=8/16

| Condition | K=16 exact | Dynamic exact | Delta | Mean width | Wide fraction |
|---|---:|---:|---:|---:|---:|
| Uniform | 82.999% | 82.990% | −0.009 pp | 14.64 | 83.0% |
| Combination holdout | 83.095% | 83.086% | −0.009 pp | 14.64 | 83.0% |
| Low edge | 98.177% | 98.168% | −0.009 pp | 15.52 | 94.0% |
| High edge | 98.320% | 98.312% | −0.009 pp | 14.95 | 86.8% |

| Hard-task metric | K=16 | Dynamic | Delta |
|---|---:|---:|---:|
| Uniform hard-task mean | 58.583% | 58.573% | −0.010 pp |
| Combination hard-task mean | 58.952% | 58.930% | −0.022 pp |

Quality was effectively preserved while the uniform probe’s mean circuit width
fell from 16 to `14.64` (about 8.5% fewer selected circuit slots). The saving
is modest because the router entropy is high and the `0.995` threshold still
sends about 83% of steps through the wide path. Low-edge examples correctly
stay mostly wide under this proxy, so entropy is not yet a useful enough
difficulty predictor for the desired large saving.

## Decision

`PROMISING PROTOTYPE — LEARNED WIDTH OPEN`.

The execution path is correct and the quality/width trade-off is safe at this
threshold, but this is not a major result and it does not justify a default
change. The next step is a small learned width head or cost predictor trained
from the measured K=8 versus K=16 loss gap, with a compute penalty and held-out
hard-task guard. The target is not to force K=8: it is to learn when the extra
eight circuits reduce error enough to pay for their cost. Threshold sweeps and
logical masks alone should not be treated as a final routing solution.

## Raw evidence and reproduction

- [K=16 ceiling JSON](diagnostic_native_active_width_20260910.json)
- [Entropy-gated K=8/16 JSON](diagnostic_native_dynamic_width_threshold0995_20260910.json)
- [K=16 configuration](../configs/ne_500m_v12_factorized_shared_routekeys_step_adapter_two_edge_mix_stable_prefix_active16.yaml)

```powershell
python audit_native_ood.py `
  --checkpoints results/checkpoints/ne500_stable_prefix_active16_s17_3000.pt `
    results/checkpoints/ne500_stable_prefix_active16_s18_3000.pt `
  --batches 24 `
  --dynamic-width-mode topk_entropy `
  --dynamic-width-min 8 `
  --dynamic-width-threshold 0.995 `
  --output results/diagnostic_native_dynamic_width_threshold0995_20260910.json
```
