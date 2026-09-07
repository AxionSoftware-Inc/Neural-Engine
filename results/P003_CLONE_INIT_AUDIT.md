# P-003 — New-bank initialization: random vs parent-cloned rows

**Decision:** `REJECTED`

A 20M parent checkpoint is expanded to the same 100M target. The random arm leaves new circuit/key rows at target initialization; the clone arm copies the top `64` parent rows with Gaussian noise `0.05`. Both arms then receive the same `5000` full-bank training steps.

| Seed | Arm | Held-out acc | In-domain acc | Held-out route NMI | Route specialization | Dead | Seen in training | Train s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | random | 78.41% | 77.81% | 0.2332 | 0.4160 | 1.69% | 7552 | 401.4 |
| 17 | clone | 78.91% | 78.07% | 0.2294 | 0.4097 | 1.54% | 7552 | 380.2 |
| 17 | **delta clone-random** | **+0.49 pp** | **+0.26 pp** | **-0.0038** | **-0.0064** | **-0.16 pp** | **+0** | — |
| 18 | random | 78.75% | 78.23% | 0.2390 | 0.4256 | 1.63% | 7552 | 378.4 |
| 18 | clone | 79.27% | 78.10% | 0.2324 | 0.4155 | 1.22% | 7552 | 387.4 |
| 18 | **delta clone-random** | **+0.52 pp** | **-0.13 pp** | **-0.0067** | **-0.0101** | **-0.41 pp** | **+0** | — |

## Interpretation

The router, active budget, optimizer, training stream, target model and number of steps are unchanged. Only initialization of the newly added bank rows differs. This is an initialization test, not a claim that cloning alone solves large-bank routing.

## Gate

- `REJECTED` — mean held-out accuracy >= +2.0 pp and no seed below -1.0 pp.
- All route coverage, active parameter and training-time diagnostics are retained in the JSON report.

## Reproduction

```bash
python benchmark_p003_clone_init.py --config configs/ne_100_v12_coverage.yaml --steps 5000 --device cuda --seeds 17 18 --clone-source-count 64 --clone-noise 0.05 --update-problems
```
