# P-003 — Training-time route exploration continuation audit

**Sana:** 2026-09-10
**Branch:** `exp/track-native-engine`
**Status:** `PILOT COMPLETE — REJECTED FOR ADOPTION`

## Gipoteza

Katta bankda yangi circuitlar kam route olgani uchun kam gradient olayotgan
bo‘lsa, training paytida hierarchical tree child’ini `5%` ehtimol bilan
tasodifiy almashtirish coverage va sparse credit’ni yaxshilashi mumkin. Bu
faqat training-time mexanizm; inference’da tabiiy hard `K=8` route saqlanadi.

## Protocol

300M seed17/18 staged 10k checkpointlaridan bir xil 1,000-step continuation
qilindi. Control `route_exploration_prob=0`, treatment `0.05`; batch `128`,
balanced task sampling, AdamW, bir xil learning rate/coverage loss/halting
loss va bir xil held-out evaluator. Har arm alohida checkpointga yozildi.

## Natija

| Seed | Arm | Held-out acc | Δ acc vs control | Held-out CE | Used circuits | Dead fraction |
|---:|---|---:|---:|---:|---:|---:|
| 17 | Control | 86.09% | — | 0.41215 | 11,045 | 51.56% |
| 17 | Exploration 5% | 85.76% | −0.34 pp | 0.40794 | 11,683 | 48.76% |
| 18 | Control | 83.93% | — | 0.45071 | 11,433 | 49.86% |
| 18 | Exploration 5% | 84.01% | +0.08 pp | 0.44614 | 11,283 | 50.51% |
| **Mean** | **Exploration − control** | — | **−0.13 pp** | **−0.00439** | — | — |

Exploration ayrim coverage/CE ko‘rsatkichlarini yaxshiladi, lekin hard accuracy
ikki seedda barqaror o‘smadi. Seed17dagi `−0.34 pp` regressiya va o‘rtacha
`−0.13 pp` foydasizligi `+2 pp` quality gate’dan ancha uzoq. Shuning uchun
faqat coverage oshgani foydali computation paydo bo‘lganini anglatmaydi.

## Qaror

`route_exploration_prob=0.05` Native Engine quality uchun `REJECTED FOR
ADOPTION`. Default route, active budget va model body o‘zgarmadi; bu variant
500M’ga scale qilinmaydi. P-003 muammosi endi route exposure’ning o‘zidan
ko‘ra circuit correctionning recurrent state/outputga ta’siri va operation
composition bilan bog‘lanishini tekshirishga o‘tadi.

## Raw evidence

- [Control seed17](runs/ne300_route_exploration_control_s17_1000.json)
- [Exploration seed17](runs/ne300_route_exploration_explore_s17_1000.json)
- [Control seed18](runs/ne300_route_exploration_control_s18_1000.json)
- [Exploration seed18](runs/ne300_route_exploration_explore_s18_1000.json)
- [Exploration config](../configs/ne_300m_v12_route_exploration_continuation.yaml)
