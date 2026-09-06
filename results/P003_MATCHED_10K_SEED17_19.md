# P-003 — Matched 10,000-step capacity continuation

Sana: 2026-09-07. Maqsad: 5,000 qadamdagi capacity screen training yetarli
bo‘lmagani uchun chalg‘itgan bo‘lishi mumkinmi, shuni tekshirish. NE-20 direct
va NE-100 progressive bir xil balanced/coverage-aware protokolda 10,000 qadam
va uchta seed bilan o‘qitildi. Model tanasi, active budget (`~1.98M`) va
evaluation protokoli o‘zgartirilmadi.

## Natijalar

| Seed | NE-20 direct | NE-100 progressive | Delta (100M − 20M) |
|---:|---:|---:|---:|
| 17 | 78.72% | 79.01% | +0.29 pp |
| 18 | 78.12% | 77.14% | -0.99 pp |
| 19 | 77.14% | 77.45% | +0.31 pp |
| **Mean** | **77.99%** | **77.86%** | **-0.13 pp** |

NE-100 progressive dead-circuit fraction seed17/18/19 bo‘yicha `5.93% / 5.38%
/ 4.59%`; NE-20 directda `0.36% / 0.36% / 0.07%`. Progressive schedule katta
bankdagi exposure muammosini biroz kamaytiradi, lekin bankning o‘zi samarali
ishlamayotganini ko‘rsatadigan dead-circuit ulushi saqlanib qoladi.

NE-100 direct 10,000-step control faqat seed17/18 da mavjud: mean `77.37%`.
Progressive schedule shu ikki seedda direct controlga nisbatan `+0.49 pp`
berdi, ammo bu foyda NE-20 bilan taqqoslaganda capacity foydasiga aylanmadi.
Schedule ta’siri ham seedga bog‘liq: seed18 NE-100 progressive NE-20 dan
`-0.99 pp` past.

## Qaror

**Capacity muammosi hal bo‘lgani tasdiqlanmadi; katta sakrash yo‘q.** 10,000
qadamda ham 100M progressive modelning uch-seed o‘rtachasi 20M modeldan
`-0.13 pp` past. Shuning uchun 300M/500M/700M/1B ga o‘tish hozircha ilmiy
asoslanmagan: avval routing/circuit utilization va seed-unstability sababini
topish kerak.

Muhim qo‘shimcha signal: NE-20 seed17 `72.16% → 78.72%` bo‘lib, 5k dan 10k
qadamga `+6.56 pp` o‘sdi. Demak avvalgi 5k screen capacityni emas,
undertrainingni ham o‘lchagan. Lekin tengroq 10k budget berilganda ham katta
model ustunligi ko‘rinmadi.

## Run ma’lumotlari

- Device: NVIDIA GeForce RTX 3060.
- NE-20 runlari taxminan 389–393 soniya, NE-100 progressive runlari taxminan
  712–713 soniya.
- Run JSONlari: `results/runs/ne20_v12_coverage_matched*_10000.json` va
  `results/runs/ne100_v12_coverage_progressive*_10000.json`.
- Configs: `configs/ne_20_v12_coverage_matched.yaml` va
  `configs/ne_100_v12_coverage_progressive.yaml`.

## Keyingi sinov

Katta modelni yana kattalashtirishdan oldin mavjud 20M/100M checkpointlarda
task-by-task route usage, dead circuits, route overlap va depth-3/compose
regressiyalarini ajratib ko‘rish kerak. Maqsad router bankni haqiqatan foydali
circuitlarga ajrata olmayaptimi yoki model body murakkab tasklarda signalni
yo‘qotyaptimi, shuni aniqlash.
