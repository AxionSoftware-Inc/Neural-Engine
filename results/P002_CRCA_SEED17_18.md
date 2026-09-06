# P-002 — Causal Responsibility Credit Assignment

**Decision:** `REJECTED`

| Seed | Arm | Held-out acc | In-domain acc | Held-out route NMI | CF NMI | CF specialization | CF +adv | Dead | Train s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | control | 60.03% | 72.68% | 0.1340 | 0.2374 | 0.3651 | 0.4427 | 0/32 | 463.3 |
| 17 | crca | 58.54% | 72.63% | 0.1473 | 0.1536 | 0.2506 | 1.3754 | 0/32 | 642.5 |
| 17 | **delta** | **-1.48 pp** | **-0.05 pp** | **+0.0133** | **-0.0839** | **-0.1145** | **+0.9328** | **+0.0/32** | — |
| 18 | control | 55.86% | 73.33% | 0.1695 | 0.2576 | 0.3637 | 0.5171 | 0/32 | 464.9 |
| 18 | crca | 57.45% | 72.34% | 0.1406 | 0.1421 | 0.2347 | 1.3802 | 1/32 | 641.6 |
| 18 | **delta** | **+1.59 pp** | **-0.99 pp** | **-0.0289** | **-0.1155** | **-0.1290** | **+0.8631** | **+1.0/32** | — |

## Gate

- Accuracy: `False` — held-out mean >= +2.0 pp and no seed worse than -1.0 pp.
- Held-out functional specialization: `False` — held-out counterfactual NMI >= +0.05, specialization >= +0.05, mean positive final-CE advantage >= +0.01, no seed advantage delta below -0.005, dead delta <= +1/32.

The router class, candidate pool and hard inference path are identical in both arms.
CRCA changes training credit only. Actual probe/gradient example overhead is recorded
in each treatment training report; inference active parameters remain the control value.

## Reproduction

```bash
python benchmark_p002_specialization.py --config configs/ne_p002_20m_32.yaml --steps 5000 --device cuda --seeds 17 18 --update-problems-on-reject
```
