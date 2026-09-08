# P-004 — Fourier/algebraic register bridge audit

Sana: 2026-09-08. Maqsad: intermediate qiymatni 64 ta yangi learned
vector sifatida emas, input encoder’da ishlatilayotgan mod-64 Fourier
koordinatalari orqali keyingi recurrent state/query’ga qaytarish composition
muammosini kamaytiradimi, tekshirish.

## Gipoteza va implementatsiya

Typed register bridge `step_logits`dan soft 64-class distribution oladi.
`register_bridge_basis="fourier"` rejimida bu distribution fixed 13-o‘lchamli
koordinatalarga (`linear + sin/cos` six modular harmonics) o‘tkaziladi va
`13 → state_dim` projection orqali keyingi query’ga beriladi. Numeric input
encoder mavjud bo‘lsa, projection uning o‘qitilgan vaznlari bilan
initializatsiya qilinadi. Shu bilan Qwen FFN yoki tayyor katta model
neyronlari ko‘chirilmaydi; faqat mavjud algebraic representation qayta
ishlatiladi.

Variant opt-in qoldi. Control va stage-only qo‘llarida bridge o‘chiq, model
body, circuit bank, active circuit budget va routing defaulti o‘zgarmadi.
Bridge arm control’dan `5,376` total/active-estimate parametr ko‘proq qo‘shadi
(`384 × 13 + 384`); fixed Fourier table parametr emas.

## Protocol

Bir xil 20M `coverage_matched_5000` seed17/18 checkpointlarida avvalgi 2×2
nazorat takrorlandi:

1. `control`: eski model / final loss;
2. `stage_only`: eski model / final + depth-2/3 composition stage loss;
3. `bridge_only`: Fourier typed bridge / final loss;
4. `bridge_stage`: Fourier typed bridge / final + stage loss.

Har bir arm bir xil batch oqimi, 2,000 continuation qadam va 1,920 misollik
held-out evaluatorni oldi. Asosiy qaror hard final accuracy va mean CE bo‘yicha
ikki seedda barqarorlik bilan qilindi.

## Natijalar

| Arm | Seed17 Δ accuracy | Seed18 Δ accuracy | Mean Δ accuracy | Mean Δ CE |
|---|---:|---:|---:|---:|
| `stage_only` | `+0.104 pp` | `+0.365 pp` | `+0.234 pp` | `−0.001328` |
| `bridge_only` | `−0.469 pp` | `−0.313 pp` | `−0.391 pp` | `+0.001305` |
| `bridge_stage` | `−0.313 pp` | `+0.365 pp` | `+0.026 pp` | `−0.000172` |

Final hard accuracy:

- seed17: control `74.8958%`, bridge-only `74.4271%`, bridge+stage `74.5833%`;
- seed18: control `75.8333%`, bridge-only `75.5208%`, bridge+stage `76.1979%`.

Depth-2/3 tasklar o‘rtachasi bridge-only’da seed17 `40.1042%` vs control
`40.3646%`, seed18 `41.4062%` vs control `41.4062%`; bridge+stage’da mos
ravishda `39.8438%` va `41.6667%`. `chain3`, `compose_add_mul` va
`state_machine` uchun barqaror umumiy siljish kuzatilmadi.

## Qaror

**REJECTED FOR ADOPTION.** Fourier/algebraic basis mavjud representationni
qayta ishlatib parametrni juda kam oshirdi, ammo intermediate qiymatni foydali
composition state’iga aylantira olmadi. Bridge-only ikki seedda ham
regressiya qildi; stage loss bilan birga o‘rtacha foyda atigi `+0.026 pp`.
Bu natija “representationni ko‘chirishning o‘zi” P-004 ceilingini
yechmasligini ko‘rsatadi.

Default model va serving path o‘zgarmadi. Kod faqat keyingi diagnostik
taqqoslashlar uchun opt-in qoldi. 700M/1B scale’ga o‘tish uchun bu auditdan
yetarli signal chiqmadi.

## Reproduction

```powershell
python -u benchmark_typed_register_bridge.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --steps 2000 --stage-loss-weight 0.1 --bridge-mode soft `
  --register-bridge-basis fourier --register-slot-count 1 `
  --register-slot-read-mode sum `
  --eval-batches 4 --eval-examples-per-task 32 --device cuda `
  --output results/runs/typed_register_fourier_2x2_ne20_seed17_seed18.json
```

Raw JSON: `results/runs/typed_register_fourier_2x2_ne20_seed17_seed18.json`.
