# P-003 — Native Engine route-fragmentation frontier recheck

**Sana:** 2026-09-10
**Branch:** `exp/track-native-engine`
**Status:** `DIAGNOSTIC COMPLETE — CAPACITY-ONLY SCALE NOT JUSTIFIED`

## Maqsad

100M, 300M va 500M Native Engine banklarida sig‘im oshganda route coverage,
tasklararo route reuse va dead-circuit ulushi qanday o‘zgarishini bir xil
held-out protokolda qayta o‘lchash. Bu yangi model o‘qitmaydi va defaultni
o‘zgartirmaydi.

## Protocol

`analyze_routes.py` har checkpoint uchun 15 taskdan 64 ta example/task,
evaluator seed `1712`, `device=cuda`, checkpointdagi evaluation split/value
range va saqlangan adaptive-inference konfiguratsiyasidan foydalandi. Jami
oltita checkpoint, 100M/300M/500M uchun seed17/18 juftliklari tekshirildi.

## Natija

| Scale | Seed | Bank | Used | Dead | Within-task Jaccard | Between-task union Jaccard | Max load |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100M | 17 | 7,552 | 5,082 (67.3%) | 32.71% | 0.0092 | 0.0488 | 0.0042 |
| 100M | 18 | 7,552 | 4,894 (64.8%) | 35.20% | 0.0093 | 0.0513 | 0.0025 |
| 300M | 17 | 22,800 | 5,809 (25.5%) | 74.52% | 0.0127 | 0.0261 | 0.0020 |
| 300M | 18 | 22,800 | 5,685 (24.9%) | 75.07% | 0.0163 | 0.0280 | 0.0021 |
| 500M | 17 | 38,600 | 9,333 (24.2%) | 75.82% | 0.0015 | 0.0126 | 0.0008 |
| 500M | 18 | 38,600 | 9,198 (23.8%) | 76.17% | 0.0016 | 0.0168 | 0.0012 |

100M→500M’da bank 5.1x kattalashgan, ammo evaluation’da ishlatilgan circuit
ulushi taxminan 66%dan 24%gacha tushgan. Dead fraction 33–35%dan 76%gacha
ko‘tarilgan va tasklararo route overlap 0.049–0.051dan 0.013–0.017gacha
pasaygan. Ikki seeddagi yo‘nalish bir xil. Within-task Jaccard juda kichik
bo‘lib qolgan; bu bir task ichida route’lar keng tarqalayotganini ko‘rsatadi.

## Talqin

Bu natija raw capacity’ning o‘zi computationga aylanmayotganini kuchli
tasdiqlaydi. Muammo faqat router logit hajmi emas: katta bankda route
coverage, circuit reuse va specialization bir-biridan uzilib ketmoqda. Shu
bilan birga bu diagnostika bankdagi qolgan circuitlar matematik jihatdan
foydasizligini isbotlamaydi — faqat joriy learned policy va evaluation
distribution ularni yetarlicha ishlatmayotganini ko‘rsatadi.

## Qaror

700M/1B capacity-only training boshlanmaydi. Keyingi Native quality sinovi
route coverage’ni yaxshilaydigan bitta minimal mexanizmni 100M/300M’da,
500M’ni scale-control sifatida tekshiradi. Active path majburan oshirilmaydi;
gate route reuse/dead fraction bilan birga held-out hard accuracy, CE va
active compute’ni talab qiladi.

## Raw evidence

- [100M seed17](runs/route_frontier_ne100_growth_from_ne20_s17_to_full_10000_20260910.json)
- [100M seed18](runs/route_frontier_ne100_growth_from_ne20_s18_to_full_10000_20260910.json)
- [300M seed17](runs/route_frontier_ne300_growth_b128_from_ne20_s17_to_full_10000_20260910.json)
- [300M seed18](runs/route_frontier_ne300_growth_b128_from_ne20_s18_to_full_10000_20260910.json)
- [500M seed17](runs/route_frontier_ne500_staged_b128_s17_full_10000_20260910.json)
- [500M seed18](runs/route_frontier_ne500_staged_b128_s18_full_10000_20260910.json)
- [Reproduction script](../analyze_routes.py)
