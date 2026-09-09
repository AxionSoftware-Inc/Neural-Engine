# P-003 Native Factorized Serial-Composition Audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Status: **rejected as a scaling fix; implementation retained opt-in**

## Question

The factorized bank normally adds the two reusable factor paths before one
nonlinearity. This test made the composition ordered and serial: factor 1
first transforms the state, then factor 2 operates on that transformed state.
It preserves reusable rows and the global router, but gives the pair an
order-sensitive interaction.

## Results

All runs use balanced batch 128, AdamW, three internal steps, adaptive
halting, and seeds 17 and 18.

| Scale | Arm | Mean accuracy | Mean CE | Params | Mean time | Peak VRAM |
|---:|---|---:|---:|---:|---:|---:|
| 300M, 3k | ordered additive | 68.451% | 0.98753 | 14.49M | 146.89s | 754 MB |
| 300M, 3k | ordered serial | 68.359% | 0.99285 | 14.49M | 146.71s | 906 MB |
| 500M, 3k | shared-slot additive | 68.281% | 0.99648 | 19.27M | 215.75s | 1,225 MB |
| 500M, 3k | ordered additive, matched | 68.073% | 1.01038 | 21.76M | 223.41s | 1,263 MB |
| 500M, 3k | ordered serial, matched | 68.138% | 1.00212 | 21.76M | 222.31s | 1,364 MB |

Per-seed 500M ordered-serial values:

- Seed17: `67.943% / 1.00512 CE`
- Seed18: `68.333% / 0.99911 CE`

At 300M, serial composition is `0.091 pp` below ordered additive and
`0.117 pp` above the shared-slot baseline, with worse CE than ordered additive.
At matched 500M it improves over ordered additive by only `0.065 pp`, but
still loses `0.143 pp` to the shared-slot baseline and costs more VRAM.

## Decision

Serial factor composition is **rejected as the current capacity/scaling fix**.
It is retained as an opt-in architecture option because it is implemented and
covered by tests, but it does not justify 700M/1B expansion. The remaining
problem is more likely virtual-address routing/assignment fragmentation than
the local algebra of adding two factor rows.

## Reproduction

```powershell
python -u train.py --config configs/ne_300m_v12_factorized_global_ordered_serial.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_300m_ordered_serial_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_300m_v12_factorized_global_ordered_serial.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_300m_ordered_serial_s18_3000 --output results/runs --seed 18
python -u train.py --config configs/ne_500m_v12_factorized_global_ordered_serial_matched.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_500m_ordered_serial_matched_s17_3000 --output results/runs --seed 17
python -u train.py --config configs/ne_500m_v12_factorized_global_ordered_serial_matched.yaml --steps 3000 --device cuda --balanced-train --log-every 1000 --run-id native_factorized_global_500m_ordered_serial_matched_s18_3000 --output results/runs --seed 18
```
