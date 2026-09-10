# P-003/P-007 native edge-coverage training audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Model: 300M ordered shared-route-key bank + rank-8 step-specific adapter

## Hypothesis

The high-edge probe uses values `[56, 63]` for every operand. Under uniform
training this joint regime is rare, especially for four-operand depth-3
tasks. The first control oversampled only the high edge and improved that
probe, but traded away low-edge and early validation quality. The final
experiment therefore mixed 25% edge examples, choosing equally between
`[0, 7]` and `[56, 63]`, while retaining 75% uniform examples.

This changes training data only; model architecture, active circuit budget,
router and step adapter remain unchanged. It is not an architecture fix.

## Matched protocol

- 300M ordered shared-route-key bank, rank-8 step adapter;
- balanced task batches, AdamW, batch 128;
- 10,000 steps, seeds 17 and 18;
- identical 24-batch evaluator for ordinary, combination-holdout, low-edge
  and high-edge probes;
- comparison against the same model and seeds trained on the uniform stream.

## Results

Values are two-seed means. Delta is two-edge mix minus the uniform-training
step-adapter control.

| Probe | Control accuracy | Two-edge mix accuracy | Delta | Control hard mean | Mix hard mean | Hard delta |
|---|---:|---:|---:|---:|---:|---:|
| uniform all | 77.266% | 77.865% | **+0.599 pp** | 46.658% | 48.926% | **+2.268 pp** |
| combination holdout | 78.264% | 78.893% | **+0.629 pp** | 47.287% | 49.555% | **+2.268 pp** |
| low edge `[0,7]` | 78.385% | 95.725% | **+17.340 pp** | 51.226% | 89.334% | **+38.108 pp** |
| high edge `[56,63]` | 72.465% | 94.562% | **+22.097 pp** | 41.276% | 86.545% | **+45.269 pp** |

The result is positive on all four probes. Dead traffic is nearly unchanged
on the ordinary stream (`20.65% → 21.22%`) and slightly lower on both edge
probes (`50.99% → 49.33%` low, `46.67% → 45.38%` high).

## Capacity check under the same recipe

The exact two-edge recipe was then run on 500M for 10,000 steps with the same
two seeds and evaluator.

| Probe | 300M mix | 500M mix | 500M − 300M | Hard delta |
|---|---:|---:|---:|---:|
| uniform all | 77.865% | 76.736% | **−1.128 pp** | **−2.192 pp** |
| combination holdout | 78.893% | 77.678% | **−1.215 pp** | **−2.908 pp** |
| low edge `[0,7]` | 95.725% | 96.011% | **+0.286 pp** | **+0.694 pp** |
| high edge `[56,63]` | 94.562% | 95.482% | **+0.920 pp** | **+2.181 pp** |

500M retains and slightly improves the edge specialization, but loses on the
ordinary and held-out-combination streams. Thus the recipe fixes distribution
coverage, not the underlying 300M→500M capacity-scaling problem.

## Depth and route diagnosis

The matched depth audit shows why the extra 500M rows did not translate into
ordinary quality. On the uniform probe, selected pair cosine falls only a
little from `0.2726 → 0.2686` at depth 2 and `0.2827 → 0.2758` at depth 3,
so selected circuits are marginally less redundant. However, the useful route
contribution relative to the query also falls from `5.67% → 5.10%` at depth 2
and `4.64% → 3.96%` at depth 3, while dead traffic rises from `57.47%` to
`61.12%`. The same direction is visible on edge probes: 500M has lower pair
cosine but weaker route/query ratios and roughly equal or higher dead traffic.

This points to a remaining active-path utilization problem, not simply a lack
of stored circuit rows. The step adapter improves the interface, and edge
coverage improves the training distribution, but 500M still spreads useful
credit too thinly across the larger bank.

For context, high-edge-only sampling reached about 97% high-edge accuracy,
but reduced low-edge accuracy by 3.711 pp and was not accepted as a general
recipe. Training longer was necessary: at 3,000 steps the two-edge mix was
only 66.23% uniform accuracy, so a short screen would have incorrectly
rejected it.

## Decision

**PROMISING OPT-IN TRAINING RECIPE; not default yet.** This is the first
controlled change that improves ordinary, held-out-combination, low-edge and
high-edge probes together. It substantially weakens the claim that the
remaining edge failure is purely a router or capacity problem: data coverage
is a major contributor.

P-003 remains active: the matched 500M run is worse on ordinary and
held-out-combination quality despite being slightly better on edge probes.
The 300M two-edge recipe is the current reference; 500M is not promoted as a
default and blind 700M/1B expansion is deferred until the scaling bottleneck
is understood.

## Reproduction

```powershell
python train.py --config configs/ne_300m_v12_factorized_shared_routekeys_step_adapter_two_edge_mix.yaml --steps 10000 --device auto --balanced-train --seed 17 --run-id ne300_step_adapter_two_edge_mix_s17_10000 --output results/runs --checkpoint results/checkpoints/ne300_step_adapter_two_edge_mix_s17_10000.pt
python train.py --config configs/ne_300m_v12_factorized_shared_routekeys_step_adapter_two_edge_mix.yaml --steps 10000 --device auto --balanced-train --seed 18 --run-id ne300_step_adapter_two_edge_mix_s18_10000 --output results/runs --checkpoint results/checkpoints/ne300_step_adapter_two_edge_mix_s18_10000.pt
python audit_native_ood.py --checkpoints results/checkpoints/ne300_step_adapter_s17_10000.pt results/checkpoints/ne300_step_adapter_s18_10000.pt results/checkpoints/ne300_step_adapter_two_edge_mix_s17_10000.pt results/checkpoints/ne300_step_adapter_two_edge_mix_s18_10000.pt --batches 24 --device auto --output results/diagnostic_native_step_adapter_vs_two_edge_mix_10000.json
```

Raw outputs:

- `results/diagnostic_native_step_adapter_vs_two_edge_mix_10000.json`
- `results/diagnostic_native_step_adapter_two_edge_mix_3000.json`
- `results/diagnostic_native_two_edge_mix_300m_500m_10000.json`
- `results/diagnostic_native_two_edge_mix_depth_300m_500m_10000.json`
- `data/generator.py` and `train.py` implement the opt-in sampler.
