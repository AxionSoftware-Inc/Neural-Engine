# P-003 Native Factorized Query-Mix Audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Status: **rejected as a consistent hard-quality fix; implementation retained opt-in**

## Question

The ordered-slot bank removes first/second slot symmetry, but its factor
combination is still static apart from a learned per-address scalar. This test
let the current state dynamically adjust the two factor weights using shared
factor gate keys. No address-local matrix parameters were added.

## Protocol

- Config: `configs/ne_300m_v12_factorized_global_ordered_querymix.yaml`
- Virtual addresses: `22,800`
- Factor rows per slot: `151`
- Query factor mix scale: `0.5`
- Steps: `1,000`
- Batch: `128`, balanced tasks, AdamW, three internal steps
- Active circuits: `8`, adaptive halting enabled
- Seeds: `17`, `18`
- Device: NVIDIA GeForce RTX 3060

The primary comparison is the ordered-slot bank without query-conditioned
mixing.

## Results

| Arm | Mean accuracy | Mean CE | Total params | Mean time | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Ordered slots, no query mix | 59.036% | 1.41467 | 14.49M | 49.31s | 754 MB |
| Ordered slots + query mix 0.5 | 59.206% | 1.41590 | 14.61M | 51.69s | 765 MB |

Per-seed query-mix values:

- Seed17: `58.776% / 1.41619 CE`
- Seed18: `59.635% / 1.41560 CE`

The seed deltas versus the ordered baseline have opposite signs (`−0.312 pp`
and `+0.651 pp`). The mean is only `+0.169 pp` over ordered slots and remains
`0.026 pp` below the shared-slot global baseline. CE is worse than ordered
slots, and the dynamic gates add about `5%` time and `11 MB` VRAM.

## Decision

Query-conditioned factor mixing is **not a consistent hard-quality fix** at
this scale and is not extended to 3,000 steps. The opt-in mechanism remains
available for future combinations, but the next test should create
combination-specific structure directly through shared parameter-free
interaction rather than only changing scalar mixture weights.

## Reproduction

```powershell
python -u train.py --config configs/ne_300m_v12_factorized_global_ordered_querymix.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_global_300m_ordered_querymix_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_300m_v12_factorized_global_ordered_querymix.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_global_300m_ordered_querymix_s18_1000 --output results/runs --seed 18
```
