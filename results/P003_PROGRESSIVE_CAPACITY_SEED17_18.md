# P-003 — Progressive capacity exposure

Sana: 2026-09-07. Maqsad: katta circuit bankini training boshida birdan to‘liq
ochish o‘rniga bosqichma-bosqich ochish capacity scaling muammosini kamaytiradimi?
Active budget, model body, circuit rank, optimizer, coverage loss va 5,000-step
training saqlandi.

Schedule:

- NE-50: capacity `1408 → 2560 → 3712` at steps `1001/2001`;
- NE-100: capacity/depth `1408/4 → 2560/4 → 4096/4 → 7552/5` at steps
  `1001/2001/3001`.

## Results against matched direct-exposure control

| Seed | Model | Direct acc | Progressive acc | Delta | Direct dead | Progressive dead |
|---:|---|---:|---:|---:|---:|---:|
| 17 | NE-50 | 71.20% | 71.30% | +0.10 pp | 6.98% | 6.49% |
| 18 | NE-50 | 71.88% | 70.91% | -0.96 pp | 5.58% | 8.00% |
| 17 | NE-100 | 71.82% | 72.03% | +0.21 pp | 6.36% | 6.21% |
| 18 | NE-100 | 71.56% | 72.16% | +0.60 pp | 6.85% | 6.53% |

## Summary

| Model | Direct mean | Progressive mean | Delta |
|---|---:|---:|---:|
| NE-50 | 71.54% | 71.11% | -0.43 pp |
| NE-100 | 71.69% | 72.10% | +0.40 pp |

Active parameters remained approximately `1.98M` in every arm. Progressive
exposure therefore does not increase inference active compute. The 100M signal
is repeatable across both tested seeds, but it is small and does not meet the
predeclared +2 pp adoption gate. NE-50 is seed-unstable and regresses on seed18.

## Decision

**REJECTED FOR DEFAULT / KEEP AS FOLLOW-UP HYPOTHESIS.** Progressive exposure
is a better lead than the one-shot warmup for NE-100, but it is not a solution:
the mean gain is only `+0.40 pp`, NE-50 gets worse, and the validation is still
the `eval_split=all` screen rather than a clean held-out generalization gate.

Next: matched 10,000-step continuation/learning curves, then only if the gain
survives, a larger-bank screen. Do not promote NE-100 or jump to 300M/500M/700M
solely from this result.

## Artifacts

Run JSONs and checkpoints are local ignored files with `ne50_v12_coverage_` or
`ne100_v12_coverage_` prefixes and `matched_5000`, `warmup_5000` or
`progressive_5000` suffixes. The schedule implementation is in `train.py` and
the opt-in configs are `configs/ne_50_v12_coverage_progressive.yaml` and
`configs/ne_100_v12_coverage_progressive.yaml`.
