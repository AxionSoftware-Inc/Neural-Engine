# P-003 native factorized bank with global routing — scale audit

## Purpose

The earlier factorized-router screen mixed two hypotheses: reusable circuit
representation and a new factorized retrieval algorithm. This audit keeps the
reusable `FactorizedMicroCircuitBank` but uses the existing hierarchical global
router, so virtual addresses are scored with the same routing family as the
independent control. The test asks whether a low-parameter factorized bank can
scale from the 100M-class to 300M/500M-class address spaces.

All arms use native V0, balanced batches, AdamW, batch 128, three internal
steps, adaptive halting, `active_circuits=8`, and 3,000 steps from scratch on
seeds 17 and 18. The factorized bank uses 87, 151, or 197 factor rows for
7,552, 22,800, or 38,600 virtual addresses. The independent controls use the
same virtual address counts and the same protocol.

## Results

| Arm | Mean accuracy | Mean CE | Total params | Mean time | Peak VRAM | Mean used virtual IDs | Factor rows used |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100M independent, 7,552 | 68.620% | 0.99017 | 100,466,025 | 215.31s | 1,946 MB | 6,968 | n/a |
| 100M factorized + global, 7,552 | 68.242% | 0.99648 | 5,884,649 | 134.98s | 656 MB | 6,896 | 87/87 |
| 300M independent, 22,800 | 68.294% | 0.98378 | 299,543,913 | 458.65s | 5,735 MB | 13,032 | n/a |
| 300M factorized + global, 22,800 | 68.932% | 0.97300 | 12,581,385 | 142.74s | 733 MB | 13,174 | 151/151 |
| 500M factorized + global, 38,600 | 68.281% | 0.99648 | 19,266,177 | 215.75s | 1,225 MB | 20,340 | 197/197 |

Per-seed values for the new factorized scale arms:

- 300M: seed17 `69.401% / 0.97207`, seed18 `68.464% / 0.97393`.
- 500M: seed17 `67.604% / 1.00422`, seed18 `68.958% / 0.98873`.

## Interpretation

- 100M → 300M factorized-global improves mean accuracy by `+0.690 pp` and
  CE by `−0.02348`. This is a positive capacity signal within the same
  factorized architecture.
- 300M → 500M then regresses by `−0.651 pp` and `+0.02347` CE. The scale law
  is therefore not monotonic; a 500M/700M/1B expansion is not justified yet.
- At 300M, factorized-global beats the direct independent control by `+0.638
  pp` and `−0.01078` CE while using `95.8%` fewer parameters, about `68.9%`
  less training time, and `87.2%` less peak VRAM.
- The factor rows themselves are not starved: all 87, 151, and 197 rows are
  used in the corresponding scale screens. The remaining problem is that
  global virtual-ID traffic becomes more fragmented as the bank grows; the
  factorized representation does not guarantee that every new combination is
  a useful primitive.
- The factorized router is materially worse than the old global router at
  3,000 steps, so the global-router form is the only factorized form retained.

## Decision

**Promising operating point, not a solved architecture.** Keep the 300M
factorized-bank + global-router configuration as an opt-in candidate for later
work. Do not make it the default and do not expand to 500M/700M/1B solely by
adding virtual addresses. P-003 remains active because the positive 300M point
does not produce a stable monotonic scaling law.

The next experiment should target virtual-ID specialization/fragmentation or a
better combination representation while preserving the successful global
router. It should not return to factorized retrieval, blind address expansion,
or forced active-path budgets.

## Reproduction

```powershell
python -u train.py --config configs/ne_100_v12_factorized_global.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_100_v12_factorized_global.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_s18_3000 --output results/runs --seed 18
python -u train.py --config configs/ne_300m_v12_factorized_global.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_300m_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_300m_v12_factorized_global.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_300m_s18_3000 --output results/runs --seed 18
python -u train.py --config configs/ne_500m_v12_factorized_global.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_500m_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_500m_v12_factorized_global.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_500m_s18_3000 --output results/runs --seed 18
```
