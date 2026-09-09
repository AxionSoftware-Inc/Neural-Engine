# P-007 — Final-loss advantage gate and task-conditioned scale audit

**Sana:** 2026-09-09  
**Status:** `REJECTED AS A QUALITY FIX; CONDITIONAL SIGNAL IS TOO UNSTABLE`  
**Checkpointlar:** 100M staged full-bank, seed17/18

## Maqsad

Oldingi diagnostikada correction ba’zi joylarda foydali, ba’zilarida zararli
bo‘lishi ko‘rindi. Shu sabab ikki arzon inference-only pilot bajarildi:

1. correction foydasini task ID asosida alohida scale bilan boshqarish;
2. final CE advantage (`CE_without_correction - CE_with_correction`) asosida
   `[step_query, circuit_delta]` feature’laridan kichik bounded gate fit qilish.

Body, router va circuit bank muzlatilgan. Hech qaysi variant defaultga
ko‘chirilmagan.

## 1. Misol darajasidagi signal

Calibration batch: 15 task × 32 misol, seed1712. Positive advantage correction
shu misolda CE’ni yaxshilaganini anglatadi.

| Checkpoint | Mean advantage | Positive fraction | Negative fraction | p10 / p90 | Advantage ↔ delta-norm Pearson |
|---|---:|---:|---:|---:|---:|
| 100M s17 | −0.002800 | 46.88% | 46.88% | −0.01403 / +0.00696 | −0.177 |
| 100M s18 | +0.001995 | 52.08% | 44.38% | −0.01302 / +0.01217 | +0.031 |

Signal aralash: correctionni hamma misolda yoqish yoki hamma misolda o‘chirish
optimal emas. Lekin oddiy correction normasi foydalilikni tushuntirmaydi —
correction normasi bilan advantage korrelyatsiyasi kuchsiz va seedga bog‘liq.

## 2. Task-conditioned scale policy

Har task uchun calibration CE bo‘yicha `{0, 0.25, 0.5, 1}` ichidan alohida
scale tanlandi. Keyin policy boshqa balanced batchda (`seed2712`) tekshirildi.

| Checkpoint | Natural scale=1 CE | Task policy CE | Δ CE | Natural acc | Policy acc | Δ acc |
|---|---:|---:|---:|---:|---:|---:|
| 100M s17 | 0.411334 | 0.416338 | +0.005004 | 85.63% | 85.63% | 0.00 pp |
| 100M s18 | 0.437267 | 0.437841 | +0.000574 | 84.79% | 84.58% | −0.21 pp |

Task-level policy calibration batchda moslashgan, lekin yangi batchda natural
scale=1ni yengmadi. Demak foydali correction faqat task identity bilan emas,
misolning aniq state/contexti bilan bog‘liq.

## 3. Feature-based advantage gate

Har bir checkpoint uchun gate faqat 100M body/router/circuit path muzlatilgan
holda fit qilindi. Training target final CE advantage’dan olindi: aniq foydali
misollarga identity gate (`scale≈1`), aniq zararli misollarga suppression
(`scale≈0`). `|advantage|>=0.001` bo‘lgan calibration misollar ishlatildi.

| Checkpoint | Calibration valid fraction | Calibration gate target accuracy | Natural eval CE | Advantage-gate eval CE | Δ CE | Δ acc |
|---|---:|---:|---:|---:|---:|---:|
| 100M s17 | 32.08% | 100% | 0.411334 | 0.416686 | +0.005352 | 0.00 pp |
| 100M s18 | 34.38% | 100% | 0.437267 | 0.438518 | +0.001251 | −0.42 pp |

Gate calibrationda targetni yodlab oldi, ammo boshqa batchda regressiya berdi.
Bu mavjud `route_bounded` gate bilan bir xil natijani boshqa objective orqali
ham qaytardi: feature’lar correction foydasini barqaror generalizatsiya qila
olmayapti yoki gate qo‘llanganda keyingi recurrent state distributioni siljiydi.

## Qaror

- Task-conditioned scale policy **rad etildi**.
- Misol darajasidagi advantage gate ham **rad etildi**.
- P-007 muammosi “bitta universal correction scale” yoki “oddiy gate yetishmasligi”
  emas. Correction path ba’zi task/misollarda signal beradi, ammo shu signal
  natural route va keyingi recurrent state bilan barqaror birlashtirilmagan.
- Keyingi ish yana inference-time scale/gate tuning emas. Circuit bankni
  final output task bilan yaxshiroq align qiladigan training signal yoki
  circuit specialization/interface o‘zgarishi kerak.

## Reproduction

Task-conditioned scale:

```powershell
python -u benchmark_p007_task_conditioned_scale.py `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s18_to_full_10000.pt `
  --device cuda --examples-per-task 32 --calibration-seed 1712 --eval-seed 2712 `
  --scales 0 0.25 0.5 1 `
  --output results/runs/p007_task_conditioned_scale_ne100_s17_s18.json
```

Advantage gate:

```powershell
python -u benchmark_p007_advantage_gate.py `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s18_to_full_10000.pt `
  --device cuda --examples-per-task 32 --calibration-seed 1712 --eval-seed 2712 `
  --fit-steps 1000 --learning-rate 0.01 --margin 0.001 `
  --output results/runs/p007_advantage_gate_ne100_s17_s18.json
```
