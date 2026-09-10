# P-003 native capacity: distribution-shift audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Checkpoints: matched 300M and 500M ordered shared-route-key models, both
seed17 and seed18, each trained for 10,000 steps.

## Purpose

The long-budget screen showed a real but small 500M gain. This audit checks
whether that gain survives a different input distribution, rather than only
the ordinary balanced validation stream.

The native generator has a fixed value vocabulary `[0, 63]`. Therefore an
artificial `[64, 127]` test would be invalid: those values have no valid input
token semantics in this model. The audit instead uses four valid probes:

1. `uniform_all`: all values `[0, 63]`, all combinations;
2. `combination_heldout`: all values, deterministic held-out combination
   bucket;
3. `low_edge_values`: values `[0, 7]`;
4. `high_edge_values`: values `[56, 63]`.

Each condition uses 24 balanced batches, 768 examples per task, and the same
hard top-k inference path. The JSON contains task/depth accuracy, routing
dead traffic and factor-row utilization for every checkpoint.

## Matched two-seed result

Accuracy is the mean over seed17 and seed18. Delta is 500M minus 300M.

| Probe | 300M accuracy | 500M accuracy | Delta | 300M hard-task mean | 500M hard-task mean | Hard delta | Dead traffic 300M → 500M |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform all | 76.797% | 77.214% | **+0.417 pp** | 45.356% | 47.190% | **+1.834 pp** | 23.80% → 12.75% |
| combination holdout | 77.609% | 78.043% | **+0.434 pp** | 45.486% | 46.452% | **+0.966 pp** | 26.37% → 14.70% |
| low edge values | 77.760% | 77.556% | **−0.204 pp** | 50.163% | 49.262% | **−0.901 pp** | 55.15% → 46.23% |
| high edge values | 73.854% | 73.364% | **−0.490 pp** | 40.256% | 39.844% | **−0.412 pp** | 52.74% → 48.05% |

Hard tasks are `reverse_sum`, `lookup`, `chain3`, `compose_add_mul`,
`compose_if`, and `state_machine`. They are reported separately because the
depth-1 tasks are already near saturation and can hide the actual architecture
problem.

## Interpretation

- 500M is not a dead-capacity result. On the ordinary stream it improves both
  accuracy and hard-task accuracy, and on combination holdout it retains the
  gain. Its larger factor bank also reduces dead traffic and uses more virtual
  addresses in the independent 10k diagnostic.
- The gain is not distribution-robust. On low/high edge probes, 500M does not
  beat 300M. Dead traffic rises sharply for both models, especially at the
  edges, which indicates that the learned route is input-distribution
  sensitive rather than a uniformly useful capacity allocator.
- The remaining quality bottleneck is concentrated in multi-step tasks. In the
  full probe, depth-2/3 mean equals the hard-task mean because those six tasks
  are exactly the depth-2/3 set; it remains below 50% even after 10k steps.
- This does not prove a fundamental impossibility of 700M/1B. It does reject
  blind capacity expansion as the next experiment: the additional rows need a
  mechanism that improves deep-task specialization and remains useful under
  routing distribution shift.

## Decision

**Status: diagnostic; P-003 remains ACTIVE.** No default or architecture was
changed. 700M/1B training is deferred. The next native experiment should be a
single controlled intervention on the depth-2/3 path, with edge-distribution
evaluation included from the start. A route-only patch should not be accepted
unless it improves hard-task accuracy and reduces dead traffic on both the
ordinary and edge probes.

## Reproduction

```powershell
python audit_native_ood.py --batches 24 --device auto
```

Raw metrics:

- `results/diagnostic_native_ood_300m_500m_10000.json`
- `audit_native_ood.py`

