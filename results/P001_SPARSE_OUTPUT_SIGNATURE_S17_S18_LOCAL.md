# Handoff D — Sparse output-signature selector

**Decision:** `REJECTED`

Dense full-bank inference is forbidden. The treatment keeps the existing retriever and circuit bank, scores only M=8 miniature signatures, then executes two real circuits.

## Quality / cost

| Seed | Control acc | Sparse acc | Δ acc | Δ CE | Mean regret red. | P95 regret red. | Recall Δ | Latency | Selector touched/decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 47.45% | 47.40% | -0.05 pp | -0.0879 | +14.5% | +19.4% | +0.0 pp | 0.994x | 3,545 |
| 18 | 46.51% | 46.46% | -0.05 pp | +0.0207 | -1.8% | +6.7% | +0.0 pp | 1.125x | 3,545 |

## Proxy component ablation

### Seed 17

| Method | Final acc | Mean regret | P95 regret |
|---|---:|---:|---:|
| key_score | 46.94% | 0.2919 | 1.4715 |
| individual_output_additive | 49.65% | 0.0641 | 0.3721 |
| joint_output_no_gru | 47.01% | 0.2658 | 1.2231 |
| joint_gru_state_norm | 47.08% | 0.2907 | 1.5370 |
| full_local_gru_head | 49.58% | 0.0608 | 0.3585 |
| sparse_signature | 47.29% | 0.2495 | 1.1863 |

### Seed 18

| Method | Final acc | Mean regret | P95 regret |
|---|---:|---:|---:|
| key_score | 47.92% | 0.2837 | 1.4646 |
| individual_output_additive | 51.32% | 0.0376 | 0.2066 |
| joint_output_no_gru | 47.43% | 0.3002 | 1.3696 |
| joint_gru_state_norm | 47.99% | 0.2875 | 1.3286 |
| full_local_gru_head | 51.32% | 0.0343 | 0.1778 |
| sparse_signature | 47.99% | 0.2889 | 1.3660 |

## Gate

- accuracy: `False` — seed17/18 mean hard accuracy >= +2 pp and neither seed negative.
- regret: `False` — mean and p95 candidate-selection regret each improve >= 10%.
- recall: `True` — isolated frozen-state candidate recall cannot decrease.
- latency: `True` — each seed sparse inference latency <= 1.25x control.
- protocol: `True` — seed17/18, 5000-step selector, E=32/M=8/active=2/T=3, no dense full-bank inference.

## Reproduction

```bash
python benchmark_p001_sparse_output_signature.py --checkpoint results\checkpoints\capacity_audit_c32_global_s17.pt --checkpoint results\checkpoints\capacity_audit_c32_global_s18.pt --device cuda --steps 5000 --update-problems-on-reject
```
