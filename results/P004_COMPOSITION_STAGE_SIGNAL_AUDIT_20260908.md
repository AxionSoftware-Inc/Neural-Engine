# P-004 — Composition stage-signal audit

Sana: 2026-09-08. Maqsad: Native Engine qiyin composition tasklarida xato
qaysi recurrent bosqichda paydo bo‘lishini o‘lchash.

## Protocol

100M, 300M va 500M staged 10k checkpointlarning seed17/18 variantlari bir xil
all-screen evaluator bilan tekshirildi. Har checkpointda 15 task × 32 misol ×
4 batch = `1,920` misol, `3` internal step va `adaptive=False` ishlatildi.
Checkpointlar `stage_loss_weight=0` bilan o‘qitilgan; hech qanday training yoki
model o‘zgarishi qilinmadi. Har recurrent step’dagi `step_logits` generator
bergan deterministic intermediate target bilan solishtirildi.

## Stage accuracy across six checkpoints

| Task | Step 0 | Step 1 | Step 2 | Final |
|---|---:|---:|---:|---:|
| reverse_sum | `1.56%` | `96.61%` | — | `96.48%` |
| chain3 | `2.99%` | `22.14%` | — | `25.26%` |
| compose_add_mul | `9.77%` | `57.81%` | `58.98%` | `58.98%` |
| compose_if | `1.82%` | `90.76%` | `91.93%` | `91.93%` |
| state_machine | `2.34%` | `1.69%` | `6.77%` | `6.77%` |

Bu yerda “final” modelning oxirgi chiqishi, intermediate ustunlar esa aynan
shu bosqichning `step_logits` accuracy’si. 64 classli chiqishda `~1.56%`
tasodifiy baseline hisoblanadi; `state_machine`ning 0/1 bosqichlari va
`chain3`ning 0-bosqichi amalda signalni o‘rganmagan.

## Talqin

- `reverse_sum` finalni deyarli to‘g‘ri topadi, lekin step-0 partial targetni
  o‘rganmaydi. Demak model ayrim vazifalarda intermediate register o‘rniga
  finalni bevosita taxmin qilmoqda.
- `compose_add_mul` va `compose_if`da birinchi partial/condition targetlar
  past, lekin keyingi step finalga yaqin. Bu ham explicit state transition
  ishlatilmayotganini ko‘rsatadi.
- `state_machine`da na chap partial, na o‘ng partial o‘rganilgan; final
  accuracy ham `6.77%` bo‘lib qolgan. Bu hozirgi composition ceilingning eng
  aniq diagnostikasi.

Shunday qilib capacity qo‘shishdan oldin intermediate state’ni foydali
register sifatida train qilish kerak. Bu router muammosi emasligini to‘liq
isbotlamaydi, ammo routerga yangi bank qo‘shishdan oldingi bottleneckni ancha
toraytiradi.

## Keyingi minimal test

Faqat depth-2/3 tasklar uchun stage loss (`stage_loss_weight=0.1`)ni yoqib,
depth-1 tasklarni auxiliary stage lossdan chiqarib tashlaydigan matched 20M
continuation qilinadi. Maqsad: composition stage signalini ko‘tarish, lekin
oldingi umumiy stage-supervision tajribasidagi depth-1 regressionni takrorlamaslik.
Control natural final loss bilan, treatment composition-only stage loss bilan
bir xil batch/step’da yuradi. Held-out overall accuracy, depth-3 accuracy va
stage accuracy birga o‘lchanadi.

**Dastlabki qaror:** stage-only training `OPEN EXPERIMENT`; 700M/1B scaling
hozircha keyingi qadam emas.

## Stage-only continuation natijasi

Yuqoridagi gipoteza 20M `coverage_matched_5000` checkpointlarida seed17/18
uchun matched 2,000-step continuation bilan tekshirildi. Control faqat final
lossdan, treatment esa faqat depth-2/3 misollarda `stage_loss_weight=0.1`
oraliq lossdan foydalandi; depth-1 misollar stage lossga kiritilmadi. Ikkala
variant bir xil batch oqimi, hard routing, optimizer va held-out evaluator
bilan yurdi. Har bir held-out o‘lchov `1,920` misoldan iborat bo‘ldi.

| Seed | Δ mean CE (treatment − control) | Δ final accuracy | Δ stage-0 | Δ stage-1 | Δ stage-2 |
|---:|---:|---:|---:|---:|---:|
| 17 | `+0.004297` | `+0.729 pp` | `+6.927 pp` | `+1.432 pp` | `+2.865 pp` |
| 18 | `+0.012997` | `−0.260 pp` | `+6.354 pp` | `+0.260 pp` | `−1.563 pp` |
| Mean | `+0.008647` | `+0.234 pp` | `+6.641 pp` | `+0.846 pp` | `+0.651 pp` |

Stage-0 accuracy ikki seedda ham aniq ko‘tarildi, ya’ni auxiliary loss
oraliq signalni kuchaytira oladi. Lekin bu signal final task quality’ga
barqaror aylanmadi: CE ikkala seedda ham yomonlashdi, final accuracy esa
qarama-qarshi bo‘lib, o‘rtacha foyda `+0.234 pp`da qoldi. Qabul qilish gate’i
`>=+2 pp` emas.

**Yakuniy qaror:** composition-only stage supervision `REJECTED FOR ADOPTION`.
U intermediate register muammosini diagnostik jihatdan tasdiqladi, ammo
oddiy auxiliary loss bilan arxitektura bottlenecki yechilmadi. Default model,
router va active budget o‘zgartirilmaydi; bu variant scale qilinmaydi.
`P-004`ning composition/dataflow muammosi ochiq qoladi. Keyingi arxitektura
sinovi lossni kattalashtirish emas, intermediate state’ni keyingi operation
uchun explicit typed register yoki operation-conditioned transition sifatida
bog‘laydigan opt-in bridge bo‘lishi kerak.

## Reproduction

```powershell
python -u diagnose_composition_stage.py `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s18_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s18_to_full_10000.pt `
  --checkpoint results/checkpoints/ne500_staged_b128_s17_full_10000.pt `
  --checkpoint results/checkpoints/ne500_staged_b128_s18_full_10000.pt `
  --batches 4 --examples-per-task 32 --split all --device cuda `
  --output results/runs/composition_stage_diagnostic_ne100_ne300_ne500_1920.json
```

Stage-only continuation:

```powershell
python -u benchmark_composition_stage_only.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --steps 2000 --stage-loss-weight 0.1 `
  --eval-batches 4 --eval-examples-per-task 32 --device cuda `
  --output results/runs/composition_stage_only_continuation_ne20_seed17_seed18.json
```
