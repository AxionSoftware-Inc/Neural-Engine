# P-003 — Frozen route-capacity clamp audit

**Sana:** 2026-09-10
**Branch:** `exp/track-native-engine`
**Status:** `DIAGNOSTIC COMPLETE — PREFIX CLAMP IS NOT A QUALITY FIX`

## Maqsad

Katta bankning o‘zi route sifatini buzayotganini yoki muammo training-time
route policy’da ekanini ajratish uchun frozen 100M/300M/500M checkpointlarda
reachable bank prefixi vaqtincha `7,552/depth-5` va `1,408/depth-4`ga qisqartirildi.
Model vaznlari, active `K=8`, adaptive halting va evaluator o‘zgartirilmadi.
Bu inference-only counterfactual; yangi checkpoint default sifatida yaratilmaydi.

## Protocol

Har scale uchun seed17/18 checkpointlari, 15 task × 64 example, evaluator seed
`1712`, CUDA va checkpointdagi adaptive inference ishlatildi. `natural` —
checkpointning o‘z routing capacity/depth’i; prefix variantlari faqat
`router.set_routing_state` orqali reachable prefixni almashtirdi.

## Natija

| Scale | Seed | Natural acc | Prefix 7,552 acc | Prefix 1,408 acc | Natural CE | Prefix 7,552 CE | Prefix 1,408 CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100M | 17 | 84.48% | 84.48% | 84.48% | 0.41446 | 0.41446 | 0.41241 |
| 100M | 18 | 84.48% | 84.48% | 84.27% | 0.43618 | 0.43618 | 0.43895 |
| 300M | 17 | 86.04% | 85.31% | 85.63% | 0.40990 | 0.40946 | 0.41270 |
| 300M | 18 | 85.00% | 85.00% | 84.48% | 0.44139 | 0.44084 | 0.44188 |
| 500M | 17 | 85.83% | 85.42% | 85.31% | 0.43320 | 0.43532 | 0.43636 |
| 500M | 18 | 84.06% | 84.06% | 84.79% | 0.45005 | 0.45021 | 0.44824 |

300M va 500M’da prefix qisqartirish ikki seedda bir xil foyda bermadi. 500M
seed18da 1,408-prefix `+0.73 pp` bergan bo‘lsa ham, seed17da `−0.52 pp` berdi;
100M va 300Mda ham yo‘nalish barqaror emas. Shuning uchun katta bankning
oddiy mavjudligi sifat regressiyasining yetarli sababi emas.

## Qaror

Prefix clamp `REJECTED AS QUALITY FIX`. Bu natija P-003 bottleneckini
“bank hajmi”dan “qaysi circuitlar trainingda route orqali foydali computationga
aylanadi” degan policy/credit muammosiga toraytiradi. Prefix clamp defaultga
qo‘yilmaydi va 700M/1B scale qarorini o‘zgartirmaydi.

## Raw evidence

- [100M seed17](runs/route_capacity_clamp_ne100_growth_from_ne20_s17_to_full_10000_20260910.json)
- [100M seed18](runs/route_capacity_clamp_ne100_growth_from_ne20_s18_to_full_10000_20260910.json)
- [300M seed17](runs/route_capacity_clamp_ne300_growth_b128_from_ne20_s17_to_full_10000_20260910.json)
- [300M seed18](runs/route_capacity_clamp_ne300_growth_b128_from_ne20_s18_to_full_10000_20260910.json)
- [500M seed17](runs/route_capacity_clamp_ne500_staged_b128_s17_full_10000_20260910.json)
- [500M seed18](runs/route_capacity_clamp_ne500_staged_b128_s18_full_10000_20260910.json)
- [Reproduction script](../benchmark_route_capacity_clamp.py)
