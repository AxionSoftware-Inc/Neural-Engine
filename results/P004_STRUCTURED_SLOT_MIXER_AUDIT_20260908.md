# P-004 — Structured multi-slot register read audit

Sana: 2026-09-08. Maqsad: ikki typed register slotini oddiy yig‘indi sifatida
o‘qish slot identifikatorini yo‘qotgani uchun composition ishlamayotgan bo‘lishi
mumkin degan gipotezani tekshirish.

## O‘zgarish

`register_slot_count=2` bilan `register_slot_read_mode="mix"` step-0 va
step-1 register contextlarini concatenation orqali kichik learned linear mixer
bilan o‘qiydi. Mixer boshlanishida ikkala slot bo‘yicha identity bloklari
qo‘yildi, shuning uchun u oldingi `slot0 + slot1` yo‘liga ekvivalentdan
boshlaydi, lekin trainingda slot identity va cross-slot interactionni
o‘rganishi mumkin. Bu faqat typed bridge faol bo‘lganda ishlaydi; circuit
selection va recurrent cell defaulti o‘zgarmadi.

Mixer 294,912 parametr qo‘shadi. Value embedding bilan birga bridge armning
total/active estimate’i controlga nisbatan 319,488 parametr ko‘proq bo‘ldi.

## Protocol

20M `coverage_matched_5000` seed17/18 checkpointlarida soft bridge,
`register_slot_count=2` va structured mixer bilan avvalgi 2×2 nazorat takrorlandi:

1. old model / final loss (`control`);
2. old model / final + depth-2/3 stage loss (`stage_only`);
3. two-slot mixer bridge / final loss (`bridge_only`);
4. two-slot mixer bridge / final + stage loss (`bridge_stage`).

Har arm bir xil batch oqimi, 2,000 continuation qadam va 1,920 misollik
held-out evaluatorni oldi.

## Natijalar

| Arm | Seed17 Δ accuracy | Seed18 Δ accuracy | Mean Δ accuracy | Mean Δ CE |
|---|---:|---:|---:|---:|
| stage_only | `+0.208 pp` | `−1.094 pp` | `−0.443 pp` | `+0.004646` |
| bridge_only | `−0.156 pp` | `−0.990 pp` | `−0.573 pp` | `+0.011609` |
| bridge_stage | `+0.313 pp` | `−0.990 pp` | `−0.339 pp` | `+0.009105` |

`bridge_stage` stage-0 o‘rtacha `+6.458 pp` yaxshilangan bo‘lsa ham stage-2
o‘rtacha `+1.172 pp` bilan adoption gate’iga yetmadi; `state_machine`
depth-3 natijasi ham barqaror yaxshilanmadi. Demak slot identityni ajratishning o‘zi intermediate
qiymatni keyingi operation uchun foydali composition state’iga aylantirmadi.

## Qaror

Structured slot mixer **REJECTED FOR ADOPTION**. U previous sum read’ni
ifodalashga qodir bo‘lsa ham ikki seedda final hard accuracy va CE bo‘yicha
regressiya berdi. Default model va active circuit budget o‘zgarmadi; patch
opt-in diagnostik kod sifatida qoldi. P-004 uchun keyingi yo‘l faqat register
layoutini ko‘paytirish emas, algebraic value/state primitive yoki circuit
specializationni alohida sinashdir. 700M/1B ga scale qilinmaydi.

## Reproduction

```powershell
python -u benchmark_typed_register_bridge.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --steps 2000 --stage-loss-weight 0.1 --bridge-mode soft `
  --register-slot-count 2 --register-slot-read-mode mix `
  --eval-batches 4 --eval-examples-per-task 32 --device cuda `
  --output results/runs/typed_register_multislot2_mix_2x2_ne20_seed17_seed18.json
```

Raw JSON: `results/runs/typed_register_multislot2_mix_2x2_ne20_seed17_seed18.json`.
