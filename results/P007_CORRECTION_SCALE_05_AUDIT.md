# P-007 — Static circuit correction scale audit

Sana: 2026-09-07. Maqsad: route replay circuit correctionni o‘zgartirganda
final output juda kam o‘zgarayotganidan kelib chiqib, correction ulushini
statik `0.5x` bilan yumshatish route sifatini yaxshilaydimi?

## Patch

`NeuralEngineV0`ga opt-in `circuit_delta_scale` qo‘shildi. Default qiymat
`1.0`, shuning uchun mavjud API va checkpointlar o‘zgarmadi. Har bir recurrent
stepda circuit delta route gain bilan birga shu scalarga ko‘paytiriladi.
`scale=0` rad qilinadi. Patch total/active parameter sonini o‘zgartirmaydi.

## Inference-only sweep

500M seed17/18 full 10k checkpointlarida held-out batchda training qilmasdan
tekshirildi:

| Scale | Seed17 accuracy | Seed18 accuracy | Izoh |
|---:|---:|---:|---|
| 0.5 | 86.67% | 83.85% | seed17da kichik test-time foyda |
| 0.75 | 86.62% | 83.85% | — |
| 1.0 | 86.25% | 83.85% | default |
| 1.5 | 85.94% | 83.75% | yomonlashdi |
| 2.0 | 85.47% | 83.44% | yomonlashdi |
| 3.0 | 85.31% | 83.39% | yomonlashdi |
| 4.0 | 85.21% | 83.23% | yomonlashdi |

Bu sweep faqat diagnostika; checkpoint scale=1.0 bilan o‘qitilganligi sababli
training foydasi deb talqin qilinmaydi.

## Training control

100M seed17 staged protocoli scale=0.5 bilan qayta bajarildi: NE-20 parent 5k
→ 1,408 clamp 5k → full 7,552 bank 5k → full continuation 5k.

| Variant | All-screen | Held-out active-8 |
|---|---:|---:|
| Default scale=1.0 | 85.57% | 86.20% |
| Scale=0.5 | 85.18% | 86.20% |

Scale=0.5 all-screenda `−0.39 pp`, held-outda esa `0.00 pp` delta berdi.
Ikki-seed acceptance gate bajarilmadi; seed18ni to‘liq trainingga kiritish
shart emas, chunki birinchi seedda quality foydasi yo‘q va held-out yutuq
ko‘rsatmadi.

## Qaror

**REJECTED FOR ADOPTION.** Static correction scale route causal muammosini
hal qilmadi. Test-time seed17dagi kichik yaxshilanish training bilan
generalizatsiya qilmadi, scale>1 esa ikki seedda yomonlashdi. `circuit_delta_scale`
APIda default-compatible opt-in diagnostika sifatida qoldi; default model
`1.0` bo‘lib qoladi.

## Reproduction

- Code: `neural_engine/model.py`, `train.py`.
- Configs: `configs/ne_100_v12_scale05_capacity_clamp_b128.yaml`,
  `configs/ne_100_v12_scale05_coverage_b128.yaml`.
- Checkpoint: `results/checkpoints/ne100_scale05_staged_s17_full_10000.pt`.
- Eval: `results/runs/ne100_scale05_active_budget_10k_s17.json`.
