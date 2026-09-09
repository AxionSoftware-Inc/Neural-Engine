# P-003 Native Factorized Product-Interaction Audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Status: **rejected as a quality fix; implementation retained opt-in**

## Question

Additive factor composition can make virtual addresses too similar. This test
added a parameter-free elementwise product of the first and second ordered
factor matrices to the composed circuit. It creates a direct `(i, j)`
interaction without a per-address parameter bank.

## Protocol and results

- Config: `configs/ne_300m_v12_factorized_global_ordered_product8.yaml`
- 300M virtual bank, `151` rows per ordered slot
- Product scale: `8.0`
- 1,000 steps, balanced batch 128, seeds 17 and 18
- Existing global router and active path unchanged

| Arm | Mean accuracy | Mean CE | Params | Mean time | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| Ordered slots, no product | 59.036% | 1.41467 | 14.49M | 49.31s | 754 MB |
| Ordered slots + product scale 8 | 58.971% | 1.41736 | 14.49M | 61.41s | 952 MB |

Per-seed product values:

- Seed17: `58.854% / 1.41146 CE`
- Seed18: `59.089% / 1.42327 CE`

The product loses `0.065 pp` mean hard accuracy, worsens CE by `0.00269`,
and raises peak VRAM by `198 MB` without adding stored parameters.

## Decision

The parameter-free product interaction is **rejected as a quality/capacity
fix**. The implementation remains opt-in for later ablations, but the next
architecture should address routing/capacity sharing rather than adding more
elementwise combination terms.

## Reproduction

```powershell
python -u train.py --config configs/ne_300m_v12_factorized_global_ordered_product8.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_global_300m_ordered_product8_s17_1000 --output results/runs --seed 17
python -u train.py --config configs/ne_300m_v12_factorized_global_ordered_product8.yaml --steps 1000 --device cuda --balanced-train --log-every 500 --run-id native_factorized_global_300m_ordered_product8_s18_1000 --output results/runs --seed 18
```
