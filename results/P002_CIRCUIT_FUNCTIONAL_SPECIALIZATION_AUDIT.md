# P-002 — Circuit functional specialization audit

Sana: 2026-09-07. Maqsad: capacity oshganda circuit bank collapse bo‘lyaptimi,
yo circuitlar bir-biridan farqli bo‘lsa ham route traffic fragmentatsiyasi
foydali specializationni yo‘qotyaptimi — shu ikki gipotezani ajratish.

## O‘lchov

100M, 300M va 500M staged 10k checkpointlar bir xil evaluator bilan tekshirildi:
har taskdan 16 ta held-out misol, 15 task, 3 internal step (`720` query-step
row). Har rowda router candidate poolidagi 32 circuit alohida ishlatilib,
chiqish vektorlari solishtirildi. Bu test inference-only; checkpointlar
o‘zgartirilmadi va natijadagi accuracy faqat diagnostik batch sanity checkidir.

## Natijalar

| Model | Candidate mean pair cosine | Selected mean pair cosine | Used circuits | Bank coverage | Used-circuit task entropy |
|---|---:|---:|---:|---:|---:|
| 100M seed17 | 0.0400 | 0.0920 | 3,442 | 45.6% | 0.245 |
| 100M seed18 | 0.0398 | 0.0971 | 3,364 | 44.5% | 0.250 |
| 300M seed17 | 0.0433 | 0.1054 | 3,676 | 16.1% | 0.142 |
| 300M seed18 | 0.0416 | 0.1007 | 3,719 | 16.3% | 0.130 |
| 500M seed17 | 0.0416 | 0.0926 | 4,854 | 12.6% | 0.067 |
| 500M seed18 | 0.0413 | 0.0935 | 4,769 | 12.4% | 0.071 |

Candidate chiqishlarining cosine’i barcha scale’da past va bir-biridan farqli;
shuning uchun “bankdagi barcha circuitlar bir xil funksiya bo‘lib qolgan” degan
gipoteza bu testda tasdiqlanmadi. 500M absolute ravishda ko‘proq circuit ishlatadi,
lekin katta bankka nisbatan route massasi juda kichik hududga jamlanadi. Task
entropy pasayishi esa circuitlar ko‘proq bitta task/domainga yopishayotganini
ko‘rsatadi.

Route composition normasi candidate individual normasi bilan solishtirganda
100M/300M/500M uchun mos ravishda taxminan `0.65–0.74` oralig‘ida bo‘ldi.
Demak tanlangan circuitlar bir-birini shunchaki nusxalamaydi, lekin ularning
combined correction’i shared encoded signalga nisbatan kichik bo‘lib qolmoqda.

## Qaror

Bu audit P-002ni yopmaydi. U muammoni “functional collapse” emas,
**over-specialization + route fragmentation** sifatida toraytiradi. P-003dagi
500M quality saturation bilan birga o‘qilganda, katta bankka yangi circuitlar
qo‘shilishi foydali umumiy primitives sonini yetarlicha oshirmayapti.

Shu sababli keyingi experiment yangi katta model train qilish emas, kichik
scale’da circuitlararo qayta foydalanishni oshiradigan opt-in bank/training
patch bo‘lishi kerak. Qabul qilish uchun faqat task entropy emas, held-out hard
accuracy, route replay sensitivity va active cost birga tekshiriladi.

## Reproduction

```powershell
$env:PYTHONPATH=(Get-Location).Path
python diagnose_circuit_specialization.py `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne500_staged_b128_s17_full_10000.pt `
  --examples-per-task 16 `
  --output results/diagnostic_circuit_specialization_scale.json
```

Seed18 natijasi `results/diagnostic_circuit_specialization_scale_seed18.json`
faylida saqlangan.
