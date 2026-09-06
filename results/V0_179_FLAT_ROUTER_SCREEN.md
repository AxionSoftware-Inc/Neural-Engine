# V0.179 — Flat full-bank router screen

## Maqsad

V0.178 auditida 32-bank uchun ikki xil headroom topildi: candidate pool ichida
ham, butun bankda ham learned route'dan yaxshiroq juftliklar bor edi. Shu sabab
hierarchical candidate retrievalni olib tashlab, barcha bank key'larini score
qiladigan, lekin executionda faqat top-2 circuitni ishlatadigan `FlatRouter`
qo‘shildi.

Bu test dense circuit execution emas. Biroq 32 key score qilish xarajati
inference budgetga alohida kiradi.

## 1000-step pilot

| Variant | Seed 17 held-out | Seed 18 held-out | Mean | Dead circuits |
|---|---:|---:|---:|---:|
| Hierarchical global reference | 25.86% | — | — | 0.0% |
| Flat, no exploration | 27.06% | 28.15% | 27.60% | 40.6% / 18.8% |
| Flat, 10% training exploration | 26.82% | 27.86% | 27.34% | 18.8% / 21.9% |

1000 qadamda flat router ikki seedda ijobiy ko‘rindi, ammo circuit usage juda
notekis bo‘ldi. Exploration dead rowsni kamaytirdi, lekin sifatni yaxshilamadi.

## 5000-step full screen

| Variant | Seed 17 held-out | Seed 18 held-out | Mean | Dead circuits |
|---|---:|---:|---:|---:|
| Hierarchical global reference | 48.18% | 46.17% | **47.18%** | 0.0% |
| Flat, no exploration | 45.26% | 44.51% | **44.89%** | 34.4% / 37.5% |

Flat router uzoqroq trainingda pilotdagi foydani saqlamadi. Full-bank key
scoring candidate retrievalni olib tashladi, lekin top-2 sparse gradient circuit
bankni yetarlicha o‘qitmadi. Shuning uchun barcha key'larni score qilishning o‘zi
capacity muammosining yechimi emas.

## Qaror

**FlatRouter default sifatida rad qilindi.** V0.178 oracle auditidagi headroom
haqiqiy, ammo uni oddiy full-bank argmax bilan olish mumkin emas. Muammo endi
uch qismga ajraldi:

1. kerakli circuit candidate poolga kirmasligi;
2. candidate ichida juftlik selection xatosi;
3. circuit/key'larning sparse training sabab foydali specializationga
   aylanishi.

Keyingi amaliy yo‘l full-bank score'ni yana kattalashtirish emas, training paytida
qo‘shimcha circuit probe qilib, kuzatilgan route alternativalari orasidagi final
task-loss farqini routerga berishdir. Bu barcha 32 circuitni har qadamda dense
ishlatmaydi, lekin probe qilingan rowsga real learning signal beradi.
