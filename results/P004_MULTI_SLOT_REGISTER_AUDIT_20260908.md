# P-004 — Multi-slot typed register bridge audit

Sana: 2026-09-08. Maqsad: oldingi typed-register bridge bitta predicted
intermediate qiymatni keyingi bosqichda overwrite qilgani uchun composition
signal yo‘qolgan bo‘lishi mumkin degan gipotezani tekshirish.

## O‘zgarish

Opt-in `register_slot_count=2` bridge step-0 qiymatini slot-0 ga, step-1
qiymatini slot-1 ga yozadi. Keyingi query ikkala slot yig‘indisini oladi;
oldingi single-slot rejimda esa har qadam yangi qiymat bilan almashtiriladi.
Register embedding zero-init bo‘lib qoladi, shuning uchun patch checkpoint
migratsiyasida default forward yo‘lini buzmaydi. Inference’da qo‘shimcha
circuit ishlatilmaydi va active circuit budget o‘zgarmaydi.

## Protocol

Oldingi auditdagi 20M `coverage_matched_5000` seed17/18 checkpointlarida
`register_slot_count=2`, soft bridge va composition stage loss bilan 2×2 arm
tekshirildi:

1. old model / final loss (`control`);
2. old model / final + depth-2/3 stage loss (`stage_only`);
3. two-slot bridge / final loss (`bridge_only`);
4. two-slot bridge / final + stage loss (`bridge_stage`).

Har arm bir xil batch oqimi va 1,920 misollik held-out evaluatorni oldi;
continuation 2,000 qadam bo‘ldi.

## Natijalar

| Arm | Seed17 Δ accuracy | Seed18 Δ accuracy | Mean Δ accuracy | Mean Δ CE |
|---|---:|---:|---:|---:|
| stage_only | `−0.208 pp` | `−0.625 pp` | `−0.417 pp` | `−0.002283` |
| bridge_only | `+0.365 pp` | `−0.833 pp` | `−0.234 pp` | `−0.001890` |
| bridge_stage | `+0.104 pp` | `−1.250 pp` | `−0.573 pp` | `+0.005282` |

Bridge-only’da seed17 yaxshilanishi seed18’da takrorlanmadi. Bridge+stage
stage-0 o‘rtacha `+6.667 pp` ko‘tarilgan bo‘lsa ham stage-2 `−0.521 pp` tushdi;
ya’ni signalni saqlashning o‘zi uni keyingi operation uchun foydali computation
ga aylantirmadi. Bu natija oldingi single-slot bridge va stage-loss auditlari
bilan bir xil xulosani mustahkamlaydi.

## Qaror

`register_slot_count=2` **REJECTED FOR ADOPTION**. Muammo faqat bitta
register qiymatining overwrite qilinishida emas; typed output signalining
representation/state-transition bilan moslashuvi va circuit specialization
ham ochiq qolmoqda. Default model, active budget va router o‘zgarmadi.
Register slotlari opt-in diagnostik kod sifatida qoldi. 700M/1B ga scale
qilinmaydi.

## Reproduction

```powershell
python -u benchmark_typed_register_bridge.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --steps 2000 --stage-loss-weight 0.1 --bridge-mode soft `
  --register-slot-count 2 --eval-batches 4 --eval-examples-per-task 32 `
  --device cuda `
  --output results/runs/typed_register_multislot2_2x2_ne20_seed17_seed18.json
```

Raw JSON: `results/runs/typed_register_multislot2_2x2_ne20_seed17_seed18.json`.
