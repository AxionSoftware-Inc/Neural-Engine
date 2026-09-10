# P-003 native active-width diagnostic

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Base: 500M stable-prefix bank, ordered factor slots, shared route keys, rank-8
step adapter, 25% low/high edge mix

## Question

Is the current fixed `active_circuits=8` budget itself limiting quality on the
hard tasks, even after stable-prefix growth has repaired the worst route
fragmentation? This is a diagnostic of the active-compute ceiling, not a final
selective-routing design.

## Protocol

- The same stable-prefix 500M stage-2 3,000-step checkpoints (seeds 17/18)
  were continued for 3,000 steps with `active_circuits=16`.
- All other model and data settings were kept unchanged.
- The evaluator used 24 batches with uniform, combination-heldout, low-edge
  `[0,7]`, and high-edge `[56,63]` conditions.
- The comparison is not compute-matched: K=16 executes twice as many selected
  circuit slots per recurrent step. It is intentionally a ceiling diagnostic.

## Result

| Condition | K=8 stable-prefix mean | K=16 mean | Delta |
|---|---:|---:|---:|
| Uniform exact accuracy | 82.040% | **82.999%** | **+0.959 pp** |
| Uniform hard-task mean | 56.695% | **58.583%** | **+1.888 pp** |
| Combination holdout | 82.357% | **83.095%** | **+0.738 pp** |
| Combination hard-task mean | 57.107% | **58.952%** | **+1.845 pp** |
| Low-edge exact accuracy | 97.617% | **98.178%** | **+0.561 pp** |
| High-edge exact accuracy | 97.522% | **98.325%** | **+0.803 pp** |
| Dead virtual-circuit fraction | 14.21% | **1.52%** | **−12.69 pp** |

The K=16 runs used the same total parameter count (`7,116,193`), but active
circuit parameters rose from about `202,768` to `405,536`; the active-parameter
estimate rose from about `1.70M` to `2.31M`. Peak VRAM was about `1,380 MB` and
the measured throughput was about `1,160 samples/s`, versus roughly `1,041 MB`
and `1,838 samples/s` for the K=8 stable-prefix run.

## Interpretation

This is the first clear positive quality signal after the capacity audits:
doubling the active width improved the difficult-task mean by about `+1.9 pp`
and reduced dead routing sharply in both seeds. That makes fixed K=8 a credible
quality bottleneck for the current 500M representation.

It does **not** prove that every sample needs 16 circuits. The experiment forced
K=16 for easy as well as hard examples, so it violates the main Neural Engine
goal of activating only the needed parameters and costs about 2x selected
circuit computation. It also does not prove that a larger total bank will scale
monotonically.

## Decision

`PROMISING DIAGNOSTIC — DYNAMIC WIDTH OPEN`.

K=16 is retained as a diagnostic ceiling and is not made the default. The next
experiment should learn a per-sample width in a bounded range such as 8–16,
with an explicit compute penalty or budget target, while preserving hard sparse
dispatch. Easy tasks should stay near K=8; hard tasks should be allowed to use
more slots when the predicted gain justifies them. The first implementation
should report logical width, selected width, quality, and actual dispatch cost
separately so a soft mask cannot be mistaken for a runtime speedup.

## Raw evidence and reproduction

- [K=8/K=16 OOD JSON](diagnostic_native_active_width_20260910.json)
- [K=16 configuration](../configs/ne_500m_v12_factorized_shared_routekeys_step_adapter_two_edge_mix_stable_prefix_active16.yaml)

The local K=16 checkpoints are intentionally not tracked in git because they
are reproducible generated artifacts:

```powershell
python train.py `
  --config configs/ne_500m_v12_factorized_shared_routekeys_step_adapter_two_edge_mix_stable_prefix_active16.yaml `
  --steps 3000 `
  --resume results/checkpoints/ne500_stable_prefix_stage2_s17_3000.pt `
  --output results/checkpoints/ne500_stable_prefix_active16_s17_3000.pt
```

Use seed 18 and the corresponding stage-2 checkpoint for the second run, then
run `audit_native_ood.py` with both K=8 and K=16 checkpoint paths to reproduce
the JSON comparison.
