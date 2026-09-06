# P-003 — Matched capacity training screen

Sana: 2026-09-07. Maqsad: oldingi 20M/50M/100M checkpointlarining turli
training protokoli noaniqligini kamaytirish. Uch bank bir xil `d_model=384`,
`rank=16`, `active_circuits=8`, `T=3`, balanced batches, AdamW, coverage loss
`0.01` va 5,000 qadam bilan mustaqil o‘qitildi. Faqat bank hajmi va unga mos
router depth o‘zgardi.

## Natijalar

Bu screen `eval_split=all` validation natijasidir; clean train/held-out
generalization gate emas.

| Seed | Model | Total params | Accuracy | CE | Used / total circuits | Dead | Active params | Active fraction | Train s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | NE-20 | 20.25M | 72.16% | 0.8607 | 1408 / 1408 | 0.00% | 1.978M | 9.77% | 195.4 |
| 17 | NE-50 | 50.33M | 71.20% | 0.8800 | 3453 / 3712 | 6.98% | 1.978M | 3.93% | 246.4 |
| 17 | NE-100 | 100.47M | 71.82% | 0.8791 | 7072 / 7552 | 6.36% | 1.981M | 1.97% | 360.8 |
| 18 | NE-20 | 20.25M | 71.95% | 0.8737 | 1403 / 1408 | 0.36% | 1.978M | 9.77% | 200.6 |
| 18 | NE-50 | 50.33M | 71.88% | 0.8714 | 3505 / 3712 | 5.58% | 1.978M | 3.93% | 246.2 |
| 18 | NE-100 | 100.47M | 71.56% | 0.8546 | 7035 / 7552 | 6.85% | 1.981M | 1.97% | 361.2 |

## Two-seed summary

| Model | Mean accuracy | Delta vs NE-20 | Mean dead fraction |
|---|---:|---:|---:|
| NE-20 | 72.06% | reference | 0.18% |
| NE-50 | 71.54% | -0.52 pp | 6.28% |
| NE-100 | 71.69% | -0.36 pp | 6.61% |

## Interpretation

The fixed active estimate stayed approximately 1.98M while stored capacity
grew 4.96x. The quality trend was not monotonic: neither 50M nor 100M beat
20M, and larger banks developed materially more dead routes. This identifies
an optimization/route-coverage bottleneck under the current protocol, not a
proof that dormant capacity is fundamentally useless.

P-003 remains `ACTIVE`. Longer matched learning curves, route-capacity
warmup/growth, and a clean held-out training split are still required before
choosing 300M, 500M, 700M or 1B.

## Reproduction artifacts

Checkpoints and run JSON files are local ignored artifacts:

- `results/checkpoints/ne20_v12_coverage_matched_5000.pt`
- `results/checkpoints/ne50_v12_coverage_matched_5000.pt`
- `results/checkpoints/ne100_v12_coverage_matched_5000.pt`
- seed18 variants with `_seed18_5000` suffix

All six runs used `train.py`, `--steps 5000`, `--device cuda` and
`--balanced-train`.
