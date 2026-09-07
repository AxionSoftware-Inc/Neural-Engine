# P-002 — Shared reusable residual bank audit

**Decision:** `REJECTED`

The independent control and shared-residual treatment start with the same router, recurrent body, output head, and per-circuit rows. The treatment adds one always-available rank-8 shared nonlinear primitive; the sparse per-circuit residual path and active circuit budget remain.

| Seed | Arm | Held-out acc | In-domain acc | Route NMI | Route specialization | CF NMI | CF specialization | CF +adv | Dead | Train s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | independent | 60.03% | 72.68% | 0.1340 | 0.1443 | 0.2374 | 0.3651 | 0.4427 | 0/32 | 433.6 |
| 17 | shared_residual | 58.31% | 72.50% | 0.1374 | 0.1420 | 0.2324 | 0.3638 | 0.4524 | 0/32 | 442.0 |
| 17 | **delta shared-independent** | **-1.72 pp** | **-0.18 pp** | **+0.0034** | **-0.0023** | **-0.0050** | **-0.0013** | **+0.0097** | **+0.0/32** | — |
| 18 | independent | 55.86% | 73.33% | 0.1695 | 0.1812 | 0.2576 | 0.3637 | 0.5171 | 0/32 | 452.2 |
| 18 | shared_residual | 56.98% | 73.67% | 0.1611 | 0.1738 | 0.2226 | 0.3524 | 0.4501 | 0/32 | 455.7 |
| 18 | **delta shared-independent** | **+1.12 pp** | **+0.34 pp** | **-0.0084** | **-0.0074** | **-0.0350** | **-0.0113** | **-0.0670** | **+0.0/32** | — |

## Gate

- Quality: `False` — held-out mean >= +2.0 pp and no seed below -1.0 pp.
- Functional specialization: `False` — counterfactual NMI >= +0.05, specialization >= +0.05, positive advantage >= +0.01, no seed below -0.005, dead delta <= 1/32.

The shared path is a small additional common primitive; it does not force an active circuit ID and does not change the router API. Active parameter accounting includes the shared rank-8 path.

## Reproduction

```bash
python benchmark_p002_shared_residual.py --independent-config configs/ne_p002_20m_32.yaml --shared-config configs/ne_p002_20m_32_shared_residual.yaml --steps 5000 --device cuda --seeds 17 18 --update-problems
```
