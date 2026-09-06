# P-004 — Cascade-consistent on-policy credit

**Decision:** `REJECTED`

Router architecture and circuit bank are unchanged. Credit target is final corrected-output CE after a one-step route change and natural suffix rerouting.

| Seed | Arm | Held-out acc | Held-out CE | On-policy acc | Mean regret | P95 regret | Suffix stability | Used circuits | Circuit grad cov | Router-key grad cov | Train s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | control | 50.42% | 2.3799 | 62.08% | 0.2903 | 1.3019 | 0.859 | 32/32 | 1.000 | 1.000 | 130.8 |
| 17 | cascade | 49.27% | 2.4047 | 61.35% | 0.2884 | 1.3257 | 0.885 | 32/32 | 1.000 | 1.000 | 170.4 |
| 17 | frozen_bank | 48.96% | 2.4714 | 63.33% | 0.0027 | 0.0081 | 0.999 | 30/32 | 0.000 | 1.000 | 158.3 |
| 17 | **cascade-control** | **-1.15 pp** | **+0.0248** | **-0.73 pp** | **-0.0019** | **+0.0238** | — | — | **+0.000** | **+0.000** | **1.30x** |
| 18 | control | 47.81% | 2.4840 | 61.67% | 0.3598 | 1.4168 | 0.844 | 32/32 | 1.000 | 1.000 | 131.8 |
| 18 | cascade | 47.71% | 2.4590 | 61.35% | 0.4113 | 1.9315 | 0.867 | 31/32 | 1.000 | 1.000 | 168.6 |
| 18 | frozen_bank | 48.65% | 2.4340 | 62.29% | 0.0030 | 0.0088 | 0.999 | 26/32 | 0.000 | 1.000 | 156.2 |
| 18 | **cascade-control** | **-0.10 pp** | **-0.0250** | **-0.31 pp** | **+0.0515** | **+0.5147** | — | — | **+0.000** | **+0.000** | **1.28x** |

## Gate

- accuracy: `False` — held-out mean >= +2 pp, on-policy mean >= +1 pp, no seed negative.
- ce: `False` — mean held-out CE delta <= -0.02 and no seed worse than +0.02.
- regret: `False` — mean and p95 one-step cascade regret each improve >= 10%.
- coverage: `True` — circuit and router-key gradient coverage do not fall by > 1/32.
- overhead: `True` — paired treatment/control training wall-clock <= 1.50x.
- prefix: `True` — counterfactual replay preserves natural prefix >= 99.9%.

## Interpretation rules

- Fixed-suffix replay is diagnostic only; it never supplies the treatment target.
- Cascade replay forces only the changed step. Every later recurrent step receives the changed state and reroutes naturally.
- `frozen_bank` is diagnostic and is not part of the adoption gate.
- A CE-only gain is insufficient; hard accuracy and regret must pass together.

## Reproduction

```bash
python benchmark_p004_cascade.py --config configs/ne_capacity_signal.yaml --steps 5000 --device cuda --seeds 17 18 --update-problems
```
