# P-001 — Uniform route-weight continuation audit

Sana: 2026-09-08. Frozen inference screen’da uniform selected-route weight
olti staged checkpointning barchasida CE’ni juda oz yaxshilagani sabab, shu
signal training continuationda ham foyda beradimi — shuni tekshirish.

## Protocol

20M `coverage_matched_5000` checkpointlarning seed17/18 variantlaridan bir xil
2,000 continuation step bajarildi. Control natural query-key top-8 softmax
weight bilan, treatment esa xuddi shu selected top-8 circuitlarga uniform
`1/K` weight bilan train qilindi. Model body, circuit bank, router selection,
batch generator, learning rate va evaluation protocol control/treatment uchun
bir xil bo‘ldi. Held-out evaluation `1,920` misol, 15 task × 32 × 4 batch;
`K=8`, 3 internal step, `adaptive=False`.

Treatment ikki xil holatda o‘lchandi:

1. `treatment_uniform`: uniform weight bilan deployment;
2. `treatment_natural`: faqat eval paytida natural softmax weightga qaytish.

## Natijalar

Delta control natural continuationga nisbatan; CE’da manfiy yaxshi.

| Seed | Treatment eval | ΔCE | Δaccuracy |
|---:|---|---:|---:|
| 17 | Uniform | `−0.002880` | `+1.094 pp` |
| 17 | Natural | `−0.002828` | `+1.146 pp` |
| 18 | Uniform | `+0.002210` | `−0.104 pp` |
| 18 | Natural | `+0.002333` | `−0.104 pp` |
| **Mean** | **Uniform** | **`−0.000335`** | **`+0.495 pp`** |
| **Mean** | **Natural** | **`−0.000247`** | **`+0.521 pp`** |

Seed17’dagi foyda natural evalda ham saqlangani treatment circuit/body
trainingga moslashganini ko‘rsatadi. Lekin seed18’da ayni continuation
regressiya berdi. Ikki seedli o‘rtacha foyda kichik, poydevor gate’dagi `+2 pp`
hard-accuracy talabidan o‘tmaydi.

## Qaror

Uniform weight bilan trainingni default recipega kiritmadim. Bu weightingning
butunlay foydasizligini emas, hozirgi model va data protokolida foyda barqaror
emasligini ko‘rsatadi. P-001ning asosiy retrieval muammosi — candidate poolga
kirmayotgan useful circuitlar — bu tajriba bilan hal qilinmadi.

**Decision:** `REJECTED FOR ADOPTION`; `route_weight_mode="natural"` default
saqlandi.

## Qayta ishlatish

```powershell
python -u benchmark_uniform_weight_continuation.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --steps 2000 --eval-batches 4 --eval-examples-per-task 32 --device cuda `
  --output results/runs/uniform_weight_continuation_ne20_seed17_seed18.json
```

Raw JSON ignored run artifact sifatida `results/runs/` ichida saqlanadi.
