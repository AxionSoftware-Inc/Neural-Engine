# P-003 — Warm-started staged bank growth (seeds 17/18/19)

Sana: 2026-09-07. Maqsad: katta bankni noldan o‘qitish o‘rniga avval kichik
reachable circuit bankda foydali computationni o‘rgatib, keyin saqlangan katta
bankni bosqichma-bosqich ochish sifatni yaxshilaydimi?

## Protocol

Har seed uchun quyidagi yo‘l bajarildi:

1. NE-20, `1408` circuit, `5,000` qadam.
2. State’ning mos parametrlarini 100M modelga ko‘chirish; faqat `1408` circuit
   va `routing_depth=4` reachable holda yana `5,000` qadam.
3. Shu checkpointdan full `7552` circuit va depth-5 routerga o‘tib yana `5,000`
   qadam.

Bank kengayganda parent circuitlar, router level’lari va umumiy engine weightlari
prefix bo‘yicha aynan ko‘chirildi; yangi bank qatorlari random initializationda
qoldi. Inference’da active budget `8` bo‘lib qoldi (`~1.98M` active params).

## Natijalar

| Seed | Warm-start clamp, 5k | Staged full-bank, 5k | NE-20 direct 10k | NE-100 direct growth 5k |
|---:|---:|---:|---:|---:|
| 17 | 77.92% | **82.99%** | 78.72% | 77.76% |
| 18 | 78.10% | **82.66%** | 78.12% | 78.54% |
| 19 | 77.16% | **81.61%** | 77.14% | — |
| **Mean** | **77.73%** | **82.42%** | **77.99%** | **78.15%** (2 seed) |

Staged full-bank variant NE-20 direct 10kdan `+4.43 pp`, noldan NE-100
progressive 10k uch-seed mean `77.86%`dan `+4.56 pp` yuqori chiqdi. Direct
warm-start growth faqat seed17/18da o‘lchangan va `78.15%` mean bo‘ldi;
staged variant shu nazoratdan `+4.67 pp` yuqori. Staged seedlar oralig‘i
`1.38 pp` bo‘lib, ijobiy signal saqlangan, lekin seed variance nol emas.

Bir xil active-budget evaluatorida clean held-out split natijasi:

| Model | Seed17 | Seed18 | Seed19 | Mean |
|---|---:|---:|---:|---:|
| NE-20 direct | 78.36% | 79.06% | 77.97% | 78.46% |
| NE-100 progressive from scratch | 78.10% | 78.52% | 78.59% | 78.40% |
| NE-100 staged growth | **82.50%** | **82.42%** | **81.48%** | **82.14%** |

Shu held-out screenda staged growth NE-20dan `+3.67 pp` yuqori. Demak
`split=all` natijadagi signal alohida held-out batchda ham saqlanadi.

## Murakkab vazifalar

| Variant | reverse_sum mean | chain3 mean | compose_add_mul mean | compose_if mean | state_machine mean |
|---|---:|---:|---:|---:|---:|
| Staged full-bank | 88.28% | 12.70% | 53.52% | 90.43% | 5.27% |
| Direct warm-start growth | 50.59% | 8.20% | 40.23% | 85.94% | 6.05% |

Eng katta sifat farqi depth-2/3 vazifalarda paydo bo‘ldi. Bu faqat oddiy
tasklar osonroq bo‘lgani uchun yuzaga kelgan umumiy accuracy o‘sishi emas.

## Route va active-budget diagnostikasi

128 ta misol/task auditida staged variantlar `6289/7552` va `6415/7552`
circuitdan foydalandi; dead ulush `16.72%` va `15.06%`. Noldan progressive
NE-100da shu audit dead ulushni `34.18% / 33.53% / 31.70%` ko‘rsatgan edi.
Staged route’ning between-task Jaccard qiymati `0.0876 / 0.0895` bo‘ldi —
task specialization saqlangan, lekin route banki noldan variantga qaraganda
kamroq fragmentlangan.

Bir xil staged checkpointda active budget sweep:

| Active circuits | Seed17 | Seed18 | Mean |
|---:|---:|---:|---:|
| 4 | 82.40% | 82.03% | 82.21% |
| 8 | 82.45% | 82.24% | 82.34% |
| 16 | 82.60% | 82.45% | 82.53% |
| 32 | 82.50% | 82.40% | 82.45% |

8 active circuit allaqachon yetarli; 16/32 ga oshirish faqat `+0.19/+0.11 pp`
atrofida qo‘shimcha beradi va active compute’ni oshiradi.

## Qaror

**PROMISING — defaultga hali qo‘yilmaydi.** Bu hozirgi tajribalar ichidagi
birinchi katta va uch seedda takrorlangan ijobiy signal; clean held-outda ham
`+3.67 pp` ustunlik saqlandi. Ammo quyidagilar hali tekshirilmagan:

- clean held-out split va aniq zero-shot composition benchmark;
- staged yo‘lning qaysi bosqichi zarur ekanini 2x2 nazorat bilan ajratish;
- 300M/500M da xuddi shu inherited-growth protokolining ishlashi.

Shuning uchun 100M staged checkpoint hozircha eng kuchli experiment sifatida
saqlanadi, lekin default branchga ko‘chirilmaydi. Keyingi rasmiy sinov staged
growthni uchinchi seedda va held-out/edge savollarda tekshirish bo‘ladi.

## Reproduction artifacts

- Warm-start utility: `expand_checkpoint.py`.
- Parent configs: `configs/ne_20_v12_coverage_matched.yaml` va
  `configs/ne_100_v12_capacity_clamp.yaml`.
- Full-bank config: `configs/ne_100_v12_coverage.yaml`.
- Run JSONlari `results/runs/ne100_clamp_warmstart_*` va
  `results/runs/ne100_growth_from_ne20_*` prefikslarida.
