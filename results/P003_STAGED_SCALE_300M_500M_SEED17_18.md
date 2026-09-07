# P-003 — Staged growth scale audit: 100M / 300M / 500M

Sana: 2026-09-07. Maqsad: oldin kichik reachable circuit bankda computationni
o‘rgatib, keyin katta bankni ochish usuli 300M va 500M scale’da ham sifatni
oshiradimi yoki foyda 100M atrofida to‘yinganmi?

## Protocol

Barcha modellar non-Transformer Neural Engine arxitekturasida, `d_model=384`,
`circuit_rank=16`, `active_circuits=8`, `batch_size=128`, balanced synthetic
task mix va bir xil optimizer/learning-rate bilan ishlatildi.

- 100M control: NE-20 parent 5k → 1,408 reachable 100M clamp 5k → full
  7,552-bank 5k → full-bank continuation 5k.
- 300M: NE-20 parent 5k → 1,408 reachable 300M clamp 5k → full 22,800-bank
  5k → full-bank continuation 5k.
- 500M: NE-20 parent 5k → inherited 100M clamp 5k → 1,408 reachable 500M
  clamp 5k → full 38,600-bank 5k → full-bank continuation 5k.

Har full continuationdan keyin checkpoint saqlandi. Seed17/18 bir xil
held-out evaluator bilan `examples_per_task=128`, `seed=1712`, active-budget
sweep `4/8/16/32` orqali tekshirildi.

## Full-screen natijalari

| Model, full exposure | Seed17 | Seed18 | Mean |
|---|---:|---:|---:|
| 100M staged, 5k | 82.99% | 82.66% | 82.82% |
| 300M staged, 5k | 83.26% | 82.58% | 82.92% |
| 500M staged, 5k | 82.92% | 82.84% | 82.88% |
| 100M staged, 10k | 85.57% | 84.19% | **84.99%** |
| 300M staged, 10k | 85.76% | 84.22% | **84.99%** |
| 500M staged, 10k | 85.83% | 84.04% | **84.94%** |

300M full 10k 100Mdan amalda farq qilmadi; 500M esa 300Mdan ham yuqori
chiqmadi. Seedlar orasidagi farq saqlanib qoldi.

## Clean held-out active-budget natijalari

Bir xil evaluatorda held-out hard accuracy:

| Model, full exposure | Active 4 | Active 8 | Active 16 | Active 32 |
|---|---:|---:|---:|---:|
| 100M staged, 10k mean | 84.77% | **84.90%** | 84.84% | 84.95% |
| 300M staged, 10k mean | 85.05% | **85.05%** | 85.16% | 85.05% |
| 500M staged, 10k mean | 84.95% | **85.05%** | 85.23% | 85.26% |

Active-8 bo‘yicha 300M va 500M aynan `85.05%` mean berdi. 500Mda active-16/32
variantlari `+0.18/+0.21 pp` qo‘shimcha ko‘rsatdi, lekin bu active compute va
latencyni oshiradi; 8 active circuit asosiy sparse rejim sifatida yetarli.

## Route va xarajat diagnostikasi

| Model | Total params | Active params/decision | Peak VRAM | Eval bank | Eval dead |
|---|---:|---:|---:|---:|---:|
| 100M | 100.47M | ~1.98M | ~1.95 GB | 7,552 | — |
| 300M | 299.54M | ~1.98M | ~5.74 GB | 22,800 | 62.87–64.21% |
| 500M | 505.83M | ~1.98M | ~9.67 GB | 38,600 | 60.26–60.97% |

300M 10k evalda `8,160–8,466` unique circuit, 500Mda `15,067–15,338`
unique circuit ishladi. 500M ko‘proq bankdan foydalansa ham uning active
budgeti va sifat natijasi 300Mdan sezilarli oshmadi. Total capacity o‘sishi
active pathga avtomatik ravishda foydali yangi computation olib kirmayapti.

## Route causality va circuit contribution diagnostikasi

10k full checkpointlarda 100% route replay ham route tanlovining final outputga
kuchli sababiy bog‘lanmaganini ko‘rsatdi. Seed17da global route swap accuracy
drop 100M/300M/500M uchun mos ravishda `-0.26/-0.52/-0.16 pp`, within-task
swap esa `-0.31/-0.05/-0.05 pp` bo‘ldi. 500M seed18da global drop `+0.05 pp`,
within-task drop `0.00 pp` bo‘ldi. Manfiy drop swap route ayrim samplelarda
naturaldan yaxshiroq bo‘lganini bildiradi; bu route’lar foydasiz degan qat’iy
isbot emas, lekin route tanlovi hali outputni boshqarmayotganini bildiradi.

Norm diagnostikasida 100M/300M/500M circuit delta’lari shared encoded
signalining taxminan `3.9–4.9%`iga teng. 500M seed17da natural route’ni global
almashtirish circuit delta farqini encoded normaning `5.2–5.9%`igacha o‘zgartirdi,
ammo hard output accuracy deyarli o‘zgarmadi. Demak route tanlovi mavjud, biroq
uning ta’siri shared input reinjection va recurrent state tomonidan
bosib ketilmoqda yoki circuitlar funksional jihatdan bir-biriga juda o‘xshash.

## Qaror

**500M scale default uchun qabul qilinmadi.** Staged growthning asosiy foydasi
real: 100M/300M/500M full stagega yetarli exposure berilganda 20M direct
controlga nisbatan katta o‘sish beradi. Ammo 100Mdan 300Mga o‘tish held-outda
faqat `+0.15 pp`, 300Mdan 500Mga esa `0.00 pp` active-8 mean bo‘ldi.

Shu sababli hozirgi bottleneck parametr soni emas, **yangi circuitlarni foydali
va qayta ishlatiladigan route’lar bilan bog‘lash hamda ularga yetarli exposure
berish** deb belgilanadi. Keyingi tajriba bankni yana kattalashtirish emas,
route fragmentation/coverage’ni kamaytiruvchi minimal, defaultni
almashtirmaydigan patch bo‘lishi kerak.

## Reproduction artifacts

- 300M configs: `configs/ne_300m_v12_capacity_clamp_b128.yaml`,
  `configs/ne_300m_v12_coverage_b128.yaml`.
- 500M configs: `configs/ne_500m_v12_capacity_clamp_b128.yaml`,
  `configs/ne_500m_v12_coverage_b128.yaml`.
- 10k full checkpoints: `results/checkpoints/ne100_growth_from_ne20_s*_to_full_10000.pt`,
  `results/checkpoints/ne300_growth_b128_from_ne20_s*_to_full_10000.pt`,
  `results/checkpoints/ne500_staged_b128_s*_full_10000.pt`.
- Active-budget JSONlar `results/runs/*staged_active_budget*10k*` nomlari bilan
  saqlandi; route auditlar `results/runs/ne300_staged_routes_10k_*` va
  `results/runs/ne500_staged_routes_10k_*` nomlari bilan saqlandi.
