# P-008 — Task-reuse routing regularizer audit

Sana: 2026-09-07. Maqsad: 100M staged checkpointda tasklar route pathlarini
mutlaqo alohida subtree’lar sifatida egallab olishini kamaytirish va shu orqali
capacity scalingni barqarorlashtirish.

## Patch

Hierarchical router har path-level soft distribution uchun tasklar orasidagi
mean variance’ni hisoblaydi. `routing_reuse_weight` bu farqni yumshoq tarzda
kamaytiradi. Circuit body, candidate pool, active budget va recurrent update
o‘zgarmadi. Regularizer faqat trainingda hisoblanadi; inference latencyga
ortiqcha loss hisoblash ta’sir qilmaydi.

## Protocol

100M seed17 full 10k checkpointdan bir xil seed/data/config bilan qo‘shimcha
2,000 continuation qadam:

| Variant | Weight | All-screen | Held-out active-8 | Used circuits | Dead fraction | Latency ms/batch |
|---|---:|---:|---:|---:|---:|---:|
| Control | 0 | 86.48% | 86.67% | 7,012 | 7.15% | 9.91 |
| Reuse | 2.0 | 86.22% | 87.03% | 5,115 | 32.27% | 10.56* |
| Reuse | 0.25 | 86.56% | 86.46% | 6,913 | 8.46% | 10.65 |

`*` dastlabki o‘lchovda inferencega training-only regularizer kirib qolganligi
sababli `22.9 ms` ko‘ringan; bu bug tuzatilib, haqiqiy qiymat `10.56 ms`ga qayta
o‘lchandi. Control bilan farq taxminan `1.07x`, sparse latency gate ichida.

Route replayda `weight=0.25` global 100% swap accuracy drop `+0.21 pp`,
within-task drop `+0.42 pp` bo‘ldi. Bu route causalityni biroz oshirdi, ammo
P-007ning `+1 pp` mezonidan o‘tmadi. `weight=2.0` uchun route causality ham
yetarli oshmadi va coverage keskin buzildi.

## Qaror

**REJECTED FOR ADOPTION.** Tasklararo reuse loss’i kerakli yo‘nalishda kichik
signal berdi, lekin hard held-out quality yaxshilanmadi. Katta weight route
collapse qildi; kichik weight esa noise darajasida va quality regression berdi.
Bu natija route fragmentation real ekanini qo‘llab-quvvatlaydi, ammo oddiy
path-distribution variance uni tuzatish uchun yetarli emasligini ko‘rsatadi.

Keyingi sinov task route’larini bir xil qilish emas, shared reusable primitive
va circuit-level credit’ni saqlab qoladigan mexanizm bo‘lishi kerak.

## Reproduction artifacts

- Code: `neural_engine/router.py`, `neural_engine/model.py`, `train.py`.
- Configs: `configs/ne_100_v12_reuse_b128.yaml`,
  `configs/ne_100_v12_reuse025_b128.yaml`.
- Checkpoints: `results/checkpoints/ne100_reuse_s17_cont2000.pt`,
  `results/checkpoints/ne100_reuse025_s17_cont2000.pt`.
- Active-budget and route audits are in `results/runs/` under matching names.
