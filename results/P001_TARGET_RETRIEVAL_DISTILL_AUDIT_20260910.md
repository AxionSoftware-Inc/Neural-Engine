# P-001 — Final-CE target retrieval distillation audit

Sana: 2026-09-10. Bu tajriba output-logit cost surrogate’dan keyingi,
retrieverga bevosita final-CE target berish gipotezasini tekshiradi.

## Gipoteza va protokol

Frozen 20M `v12_coverage_matched` checkpointlardan one-swap probe bilan har bir
query uchun target circuit topildi. Target qidiruvi ikki manbani birlashtirdi:

1. tabiiy `candidate_pool=32` ichidagi barcha circuitlar;
2. full reachable bank bo‘yicha mavjud key score top-8 circuitlari.

Har bir candidate haqiqiy final CE orqali tekshirildi. Eng yaxshi circuitning
contiguous `active_circuits=8` group base’i hierarchical router uchun teacher
target qilindi. Keyin faqat router `level_projections`, `level_bias` va `keys`
1000 qadam o‘qitildi. Circuit banki, recurrent body, correction va hard
inference budget muzlatildi.

Bu exhaustive subset oracle emas: teacher one-swap + full-key-top-8 probe bilan
cheklangan. Maqsad oracle’ni production route sifatida ishlatish emas, balki
final-target retrieval signalining o‘zi mavjud retrieverga ko‘cha oladimi,
degan savolni ajratish edi.

## Natijalar

| Seed | Target gain CE | External target fraction | Baseline CE / acc | Treatment CE / acc | ΔCE | Δaccuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.018019 | 70.69% | 0.809630 / 75.83% | 0.829784 / 75.42% | +0.020154 | −0.42 pp |
| 18 | 0.015981 | 71.04% | 0.805155 / 75.83% | 0.819587 / 73.75% | +0.014433 | −2.08 pp |
| **Mean** | **0.017000** | **70.87%** | — | — | **+0.017293** | **−1.25 pp** |

Target teacherning o‘zi real headroom ko‘rsatdi: 92% atrofidagi rows natural
route’dan yaxshiroq one-swap alternative topdi, va targetlarning ~71% current
candidate pooldan tashqarida edi. Ammo target loss trainingdan keyin held-out
natural route sifati ikki seedda ham yomonlashdi. Selected circuit coverage ham
seed17da `1227 → 1067`, seed18da `1244 → 1006` ga qisqardi.

## Qaror

**REJECTED FOR ADOPTION.** Final-CE one-swap targetni frozen natural querylar
ustida offline hierarchical tree/key lossga distill qilish retrieval gapni
yopmadi. Bu natija useful circuits yo‘qligini emas, teacher target va router
o‘zgargandan keyingi query/state distribution o‘rtasida mismatch borligini
ko‘rsatadi. Offline targetlar retrieverni target group’lariga yig‘ib, tabiiy
cascade taqsimotini buzdi.

Shu sabab quyidagilarni takrorlamaymiz:

- faqat frozen querylar bilan router key/tree’ni qayta o‘qitish;
- one-swap targetni doimiy group target sifatida production defaultga kiritish;
- shu recipe’ni 300M/500Mga scale qilish.

P-001 ochiq qoladi. Keyingi jiddiy yo‘l — router o‘z yaratgan on-policy
query/state taqsimotida qayta calibration qilinadigan yoki body va retrieval
signalini birga o‘rgatadigan end-to-end variant. U ham avval 20M seed17/18da
candidate recall, held-out regret, hard accuracy va active budget bilan
tekshiriladi.

## Reproduksiya

```powershell
python benchmark_target_retrieval_distill.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --train-batches 2 --eval-batches 1 --examples-per-task 16 `
  --global-topk 8 --steps 1000 --batch-size 2048 `
  --learning-rate 3e-4 --device cuda `
  --output results/runs/target_retrieval_distill_ne20_s17.json

python benchmark_target_retrieval_distill.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --train-batches 2 --eval-batches 1 --examples-per-task 16 `
  --global-topk 8 --steps 1000 --batch-size 2048 `
  --learning-rate 3e-4 --device cuda `
  --output results/runs/target_retrieval_distill_ne20_s18.json
```

Artefakt: `benchmark_target_retrieval_distill.py`.
