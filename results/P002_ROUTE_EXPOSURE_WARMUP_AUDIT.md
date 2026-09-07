# P-002 — Route-exposure warmup audit

**Decision:** `REJECTED`

The treatment uses fixed task-stable routes only for the first `1000` training steps, then switches to the unchanged learned hard router. The control learns routes from step 1.

| Seed | Arm | Held-out acc | In-domain acc | Held-out route NMI | Route specialization | CF NMI | CF specialization | Dead | Train s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | control | 60.03% | 72.68% | 0.1340 | 0.1443 | 0.2374 | 0.3651 | 0/32 | 470.6 |
| 17 | exposure | 59.79% | 73.46% | 0.1514 | 0.1586 | 0.2219 | 0.3481 | 0/32 | 473.7 |
| 17 | **delta** | **-0.23 pp** | **+0.78 pp** | **+0.0173** | **+0.0143** | **-0.0155** | **-0.0170** | **+0.0/32** | — |
| 18 | control | 55.86% | 73.33% | 0.1695 | 0.1812 | 0.2576 | 0.3637 | 0/32 | 472.8 |
| 18 | exposure | 57.63% | 73.26% | 0.1887 | 0.1985 | 0.2261 | 0.3459 | 0/32 | 459.0 |
| 18 | **delta** | **+1.77 pp** | **-0.08 pp** | **+0.0192** | **+0.0173** | **-0.0315** | **-0.0178** | **+0.0/32** | — |

## Gate

- Accuracy: `False` — held-out mean >= +2.0 pp and no seed worse than -1.0 pp.
- Specialization: `False` — route and counterfactual NMI/specialization each mean >= +0.05, no held-out accuracy seed below -1.0 pp, and no dead-fraction increase.

The router class, candidate pool, circuit body, recurrent state update, active circuit count, optimizer, data stream, and total steps are paired. The only treatment difference is the initial task-stable route exposure.
Training gradient exposure is recorded per sampled circuit; inference active parameters remain exactly the same in both arms.

## Interpretation

A `PROMISING` result is evidence for a follow-up schedule study, not an automatic default change. A `REJECTED` result closes this simple warmup recipe while leaving more structured circuit initialization or credit assignment as separate hypotheses.

## Reproduction

```bash
python benchmark_p002_route_exposure.py --config configs/ne_p002_20m_32.yaml --steps 5000 --warmup-steps 1000 --device cuda --seeds 17 18 --update-problems
```
