# P-001 — Final output-logit cost-router screen

Sana: 2026-09-10. Bu screen oldingi query/key cost surrogate va random
output-signature variantlaridan keyingi kichik diagnostik tajriba bo‘ldi.
Maqsad — candidate circuitning query-dependent chiqishini mavjud output head
orqali ko‘rsatish final corrected CE costni post-hoc surrogate uchun yaxshiroq
ifodalay oladimi, degan gipotezani tekshirish.

## Protokol

- Frozen 20M `v12_coverage_matched` checkpointlar, seed17 va seed18;
- `candidate_pool=32`, `active_circuits=8`, `internal_steps=3`;
- train: 3 batch × 16 example/task;
- held-out: 1 batch × 16 example/task;
- calibration: 1,000 AdamW qadam, batch size 4,096;
- label: har bir candidate uchun one-swap final CE;
- feature: query, candidate key, selected-route summary, key score, internal
  step va candidate circuit outputidan `model.output(...)` logits;
- expert/circuit bank, router defaulti va inference API muzlatilgan.

Output-logit feature’ni hisoblash uchun calibration paytida 32 ta candidate
circuitning chiqishi quriladi. Shuning uchun bu feature active `K=8` budgetga
mos keladigan production route emas; u faqat utility signalining o‘zini
diagnostika qiladi.

## Natija

| Seed | Natural CE | Local oracle CE | Predicted route CE | Surrogate ΔCE | Oracle recovery | Top-1 match |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.809630 | 0.801241 | 0.809958 | −0.000328 | −3.91% | 3.33% |
| 18 | 0.805155 | 0.798001 | 0.805625 | −0.000470 | −6.58% | 2.64% |
| **Mean** | — | — | — | **−0.000399** | **−5.25%** | **2.99%** |

Calibration Smooth-L1 loss seed17/18da mos ravishda `0.02410/0.02542` bo‘ldi,
ammo held-out route selection yaxshilanmadi. Ikkala seedda ham predicted route
natural route’dan yomonroq chiqdi; bu feature final-cost rankingni
generalizatsiya qilmaganini bildiradi.

## Qaror

**REJECTED FOR ADOPTION.** Candidate outputni final logitsga proyeksiya qilish
frozen post-hoc surrogate uchun yetarli utility signal bermadi. Natija oldingi
query/key MLP, random output signature va offline key retraining screenlari
bilan bir xil yo‘nalishda: calibration loss pasayishi production route
qualityiga aylanmadi.

Bu natija quyidagilarni isbotlamaydi:

- candidate output information-theoretic jihatdan foydasizligini;
- end-to-end retrieval va selector birgalikda o‘qitilsa foyda chiqmasligini;
- circuit composition yoki state interface muammosi yo‘qligini.

U faqat hozirgi frozen bank + natural route ustida, 32-candidate one-swap
label bilan o‘qitilgan post-hoc MLP direct fix emasligini ko‘rsatadi. P-001
ochiq qoladi. Keyingi Native Engine tajribasi yana feature qo‘shish emas,
candidate retrieval/selection signalini modelning o‘qitilish jarayoniga
end-to-end va active-budgetni saqlagan holda ulashga qaratiladi.

## Reproduksiya

```powershell
python benchmark_route_cost_surrogate.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --train-batches 3 --eval-batches 1 --examples-per-task 16 `
  --surrogate-steps 1000 --surrogate-batch-size 4096 `
  --signature-mode output_logits --device cuda `
  --output results/runs/route_cost_output_logits_ne20_s17.json

python benchmark_route_cost_surrogate.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --train-batches 3 --eval-batches 1 --examples-per-task 16 `
  --surrogate-steps 1000 --surrogate-batch-size 4096 `
  --signature-mode output_logits --device cuda `
  --output results/runs/route_cost_output_logits_ne20_s18.json
```

Artefakt: `benchmark_route_cost_surrogate.py` (`--signature-mode
output_logits`).
