# P-003 Native Factorized Global Pair-Basis Audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Status: **rejected as a hard-quality improvement; implementation retained opt-in**

## Question

The factorized bank composes each virtual circuit from two reusable factor rows.
This experiment added a shared rank-4 interaction basis for the pair while
keeping the existing hierarchical global router. The earlier pair experiment
used the factorized router, so this isolates the representation change from
the retrieval change.

## Protocol

- Config: `configs/ne_300m_v12_factorized_global_pair4.yaml`
- Virtual addresses: `22,800`
- Factor rows: `151`
- Pair basis rank: `4`
- Steps: `1,000` screen and `3,000` continuation
- Batch: `128`, balanced tasks, AdamW, three internal steps
- Active circuits: `8`, adaptive halting enabled
- Seeds: `17`, `18`
- Device: NVIDIA GeForce RTX 3060

The no-pair baseline is the 300M factorized-global configuration with the same
151 factor rows and global router.

## Results

| Steps | Arm | Mean accuracy | Mean CE | Total params | Mean time | Peak VRAM |
|---:|---|---:|---:|---:|---:|---:|
| 1,000 | no pair basis | 59.232% | 1.41985 | 12.58M | 48.06s | 733 MB |
| 1,000 | shared pair rank 4 | 58.919% | 1.41765 | 12.63M | 54.20s | 734 MB |
| 3,000 | no pair basis | 68.242% | 0.99648 | 12.58M | 142.74s | 733 MB |
| 3,000 | shared pair rank 4 | 68.125% | 0.99293 | 12.63M | 162.66s | 734 MB |

Per-seed 3,000-step pair values:

- Seed17: `67.969% / 0.99740 CE`, `162.19s`
- Seed18: `68.281% / 0.98846 CE`, `163.12s`

At 1,000 steps the pair basis loses `0.313 pp` hard accuracy but improves CE
by about `0.00220`. After 3,000 steps it still loses `0.117 pp` hard accuracy,
although CE is `0.00355` lower. The hard-quality result is not a reliable
capacity gain, and the added interaction path is about `14%` slower at 3,000
steps.

## Decision

The shared rank-4 pair basis is **rejected as the current quality/capacity
fix**. It remains an opt-in representation component for future combinations,
but it does not solve the non-monotonic 300M→500M scaling problem.

The next representation experiment should remove the stronger symmetry in the
current factor bank by using separate reusable factor tables for the first and
second address slots. This keeps sharing and the successful global router,
without adding an independent parameter island for every address.

## Reproduction

```powershell
python -u train.py --config configs/ne_300m_v12_factorized_global_pair4.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_300m_pair4_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_300m_v12_factorized_global_pair4.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_300m_pair4_s18_3000 --output results/runs --seed 18
```

Artifacts:

- `results/runs/native_factorized_global_300m_pair4_s17_3000.json`
- `results/runs/native_factorized_global_300m_pair4_s18_3000.json`
- `configs/ne_300m_v12_factorized_global_pair4.yaml`
