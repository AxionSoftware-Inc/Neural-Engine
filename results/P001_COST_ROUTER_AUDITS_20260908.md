# P-001 — Cost-router feature and key-training audit

Sana: 2026-09-08. Bu hujjat route-neighborhood auditida topilgan target-cost
headroomni amaliy, arzon routerga aylantirish uchun ketma-ket tekshirilgan
variantlarni jamlaydi.

## 1. Query/key cost surrogate

20M seed17/18 frozen checkpointlarda train splitdan `92,160` one-swap final CE
label yig‘ilib, query + candidate key + selected-route summary + key-score +
step identity bilan `261,705` parametrli MLP o‘qitildi. Held-out natija:

| Seed | Natural CE | Local oracle CE | Surrogate CE | ΔCE | Oracle recovery |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.818498 | 0.808923 | 0.818857 | −0.000359 | −3.75% |
| 18 | 0.818752 | 0.809788 | 0.818136 | +0.000616 | 6.87% |

Calibration loss yaxshi tushgan bo‘lsa ham, held-out route generalizatsiyasi
yo‘q. Bu feature set final output costni yetarli ifodalamaydi.

## 2. Candidate output signature

Surrogate’ga candidate circuitning query-dependent outputini fixed random
sketch sifatida qo‘shish sinab ko‘rildi. 1,440 held-out step-sample bo‘yicha
natijalar:

| Signature | Seed17 ΔCE | Seed18 ΔCE | O‘rtacha ΔCE | Oracle recovery |
|---:|---:|---:|---:|---:|
| 8-D | +0.000209 | −0.000191 | +0.000009 | ~0.1% |
| 32-D | +0.000352 | +0.000462 | +0.000407 | ~4.3% |
| 64-D | +0.000617 | +0.000554 | +0.000585 | ~6.3% |

64-D variantning batch 480 runtime screeni natural stats yo‘liga nisbatan
`17.60 ms → 55.56 ms`, ya’ni `+37.96 ms` va `3.16x` total overhead berdi.
Sabab: 32 ta candidate circuitning signature’i hisoblanadi, holbuki model
faqat `K=8` circuitni ishlatishi kerak. Kichik quality signal active-budget
tamoyilini buzadigan xarajatni oqlamaydi.

## 3. Existing key’larni final-cost label bilan qayta o‘qitish

Faqat `router.keys` train qilindi; circuit bank, recurrent body, tree
projections va inference API muzlatildi. Candidate pool va active `K=8`
o‘zgarmadi. Target soft-label — frozen model one-swap final CE.

| Variant | Seed | Key update rel. L2 | ΔCE | Δaccuracy |
|---|---:|---:|---:|---:|
| LR `2e-3`, anchor `.01` | 17 | 0.960 | +0.003012 | 0.000 pp |
| LR `2e-3`, anchor `.01` | 18 | 0.999 | +0.005057 | −0.625 pp |
| LR `2e-4`, anchor `.1` | 17 | 0.780 | +0.002834 | 0.000 pp |
| LR `2e-4`, anchor `.1` | 18 | 0.769 | +0.005716 | −0.417 pp |

Kichik LR va kuchli anchor driftni biroz kamaytirdi, lekin quality
regressiyasi saqlanib qoldi. Offline one-swap labels tabiiy route key
geometriyasiga oddiy softmax-ranking sifatida ko‘chmayapti.

## Yakuniy qaror

Quyidagi yo‘llar **direct fix sifatida rad qilindi**:

- candidate poolni 32→64/128/256 ga oddiy kengaytirish;
- full-bank raw key top-k route;
- query/key/route-summary final-cost MLP;
- 8/32/64-D output-signature surrogate;
- mavjud key’larni offline one-swap CE label bilan qayta o‘qitish.

Shu bilan birga P-001 yopilmadi. Route-neighborhood probe real retrieval
headroom borligini, ayniqsa 500M’da, ko‘rsatdi. Muammo endi aniqroq:
candidate retrieval va selection target final outputga moslashmagan, ammo
uni arzon key yoki oddiy signature bilan post-hoc tuzatib bo‘lmaydi.

Keyingi arxitektura uchun talablar:

1. route utility final corrected output costga bog‘langan bo‘lsin;
2. candidate retrieval va subset selection birgalikda o‘qilsin;
3. feature hisoblash `K=8` active budgetni yashirincha 32 ga aylantirmasin;
4. held-out hard accuracy va CE ikki seedda tekshirilsin.

Hozircha bankni 700M/1Bga oshirishga asos yo‘q; avval shu utility signalni
20M’da ishonchli qilib olish kerak.

Artefaktlar: `benchmark_route_cost_surrogate.py`,
`train_cost_aligned_keys.py`, `analyze_route_neighborhood.py`.
