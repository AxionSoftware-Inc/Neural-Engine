# P-003 native dynamic-width oracle audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Checkpoint: 500M stable-prefix K=16, seeds 17/18

## Question

Is there enough per-example quality difference between K=8 and K=16 to justify
training a real width predictor? Before adding a predictor, this audit measures
an oracle upper bound.

## Protocol

- The same K=16 checkpoint was instantiated twice, once with fixed K=8 and once
  with fixed K=16. The learned weights and input batches were identical.
- For every example, both final logits were computed. The oracle selected the
  lower per-example cross-entropy after adding `lambda * width_fraction`, where
  K=8 costs `0.5` and K=16 costs `1.0`.
- The oracle therefore computes both widths and is not deployable. It only
  answers whether a learned selector has useful headroom.
- Two seeds, 24 batches per condition, uniform, combination-heldout, low-edge,
  and high-edge probes were used.

## Result: two-seed means

| Lambda | Uniform exact | Uniform hard mean | Uniform active width | Combination exact | Combination hard mean |
|---:|---:|---:|---:|---:|---:|
| 0.01 | **83.090%** | **58.757%** | 55.1% | **83.199%** | **59.147%** |
| 0.03 | 83.064% | 58.691% | 52.2% | 83.177% | 59.093% |
| 0.05 | 83.060% | 58.681% | 51.3% | 83.168% | 59.071% |
| 0.10 | 83.047% | 58.670% | **50.6%** | 83.155% | 59.049% |

For reference, fixed K=16 uniform exact accuracy was `82.999%` and hard-task
mean `58.583%`; fixed K=8 was `83.008%` and `58.664%` on the same two-seed
protocol. The oracle retained a small quality advantage while approaching half
the active width. Edge probes showed the same pattern: at `lambda=0.05`, active
width was about 50.6–50.7% and exact accuracy stayed around `98.2–98.4%`.

## Interpretation

This is a positive feasibility signal. The model contains examples where the
extra eight circuits do not improve the answer, and a smaller subset can be
used without quality loss. It also shows that the weak entropy proxy was the
limitation: entropy-gating saved only about 8.5% width, while the teacher oracle
could save about 49% under a moderate penalty.

The result is not evidence that a deployable router will achieve the oracle
curve. The selector must predict the K=8 versus K=16 loss gap from the current
state before executing the wider route, and the recurrent state can diverge
after a narrow choice. The oracle is therefore a gate for the next experiment,
not a model result to ship.

## Decision

`POSITIVE ORACLE HEADROOM — LEARNED WIDTH PREDICTOR JUSTIFIED`.

Next, train a small width head from paired K=8/K=16 loss-gap labels on held-out
calibration examples. Use a cost-aware target around `lambda=0.05`, keep K=16
as a ceiling, and evaluate on a disjoint seed with actual one-path dispatch.
Report predictor accuracy, hard-task regret, mean executed width, and measured
runtime separately from the oracle.

## Raw evidence and reproduction

- [Oracle JSON](diagnostic_native_dynamic_width_oracle_20260910.json)
- [Oracle benchmark](../benchmark_native_dynamic_width_oracle.py)
- [K=16 configuration](../configs/ne_500m_v12_factorized_shared_routekeys_step_adapter_two_edge_mix_stable_prefix_active16.yaml)

```powershell
python benchmark_native_dynamic_width_oracle.py `
  --checkpoints results/checkpoints/ne500_stable_prefix_active16_s17_3000.pt `
    results/checkpoints/ne500_stable_prefix_active16_s18_3000.pt `
  --batches 24 `
  --lambdas 0.01 0.03 0.05 0.10 `
  --output results/diagnostic_native_dynamic_width_oracle_20260910.json
```
