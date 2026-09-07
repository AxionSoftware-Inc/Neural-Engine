# P-001 — Query/key final-cost surrogate audit

Sana: 2026-09-08. Maqsad: one-swap auditda ko‘ringan local selection regretni
targetni inference’da ko‘rmasdan, yengil trainable cost surrogate bilan
qaytara olish mumkinmi — shuni tekshirish.

## Protocol

20M `coverage_matched_5000` Native checkpointlar seed17/18 uchun circuit bank
va model body muzlatildi. Train splitdan 4 ta balanced batch, held-out splitdan
2 ta balanced batch, har taskdan 16 misol olindi. Har bir internal step va 32
candidate circuit uchun frozen model one-swap final CE labeli yig‘ildi:

- train calibration: `92,160` candidate-step samples;
- held-out evaluation: `1,440` candidate-step samples;
- active budget `K=8`, candidate pool `32`, internal steps `3`;
- surrogate: `261,705` parameterli kichik MLP, `1,000` AdamW update.

Surrogate feature’lari: recurrent query, candidate key, natural selected-route
mean key, query–candidate key score va step identity. Held-out inference’da
surrogate eng arzon deb topgan candidate bilan natural route’dagi eng
kam-weight circuit almashtirildi; natija yana frozen modelning haqiqiy final
CE’si bilan o‘lchandi.

## Natijalar

| Seed | Natural CE | Local one-swap oracle CE | Surrogate route CE | Surrogate ΔCE | Oracle gain recovery | Oracle top-1 match |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.818498 | 0.808923 | 0.818857 | −0.000359 | −3.75% | 3.47% |
| 18 | 0.818752 | 0.809788 | 0.818136 | +0.000616 | 6.87% | 2.92% |

Seed17’da surrogate natural route’dan biroz yomonroq chiqdi; seed18’da juda
kichik foyda berdi. Ikki seedning o‘rtacha surrogate delta’si taxminan
`+0.000129 CE`, ya’ni amaliy sifat yutug‘i yo‘q. Calibration smooth-L1 loss
`0.02957/0.03289` gacha tushgan bo‘lsa ham, held-out route tanlovi foydali
bo‘lmadi. Bu feature’lar final output costni yetarli ifodalamayotganini
ko‘rsatadi.

## Qaror

**Hozirgi query + key + route-summary surrogate rad qilindi.** Bu target-cost
router g‘oyasi umuman imkonsiz degani emas; aynan shu arzon feature set
circuitning query-dependent outputini ko‘rmaydi va final CE’ni generalizatsiya
qila olmadi.

Keyingi minimal tekshiruv — candidate circuitdan kichik output signature’ni
featurega qo‘shish. Lekin u 32 candidate circuitni hisoblash orqali active
budgetni yashirincha oshirishi mumkin. Shuning uchun avval signature’ning CE
correlation va qo‘shimcha routing cost’i alohida o‘lchanadi; foydali signal
bo‘lmasa, full routerga integratsiya qilinmaydi.

## Artefakt

- Script: `benchmark_route_cost_surrogate.py`.
- Raw runs: `results/runs/route_cost_surrogate_ne20_s17.json` va
  `results/runs/route_cost_surrogate_ne20_s18.json`.
