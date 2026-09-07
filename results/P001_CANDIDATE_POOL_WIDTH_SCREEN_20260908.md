# P-001 — Candidate-pool width screen

Sana: 2026-09-08. Maqsad: 32 ta circuitlik lokal candidate oynasi kerakli
circuitlarni yashirayotgan bo‘lsa, frozen checkpointda oynani kengaytirish
sifatni izchil yaxshilaydimi — shuni ajratish.

## Protocol

100M, 300M va 500M staged 10k checkpointlarning seed17/18 variantlari bir xil
held-out evaluator bilan tekshirildi. Har checkpoint uchun bir xil 1,920 ta
misol ishlatildi: 15 task × 32 misol × 4 batch. Circuit bank, router weight,
query va model body o‘zgarmadi; faqat hierarchical routerdagi
`candidate_pool` inference vaqtida `32, 64, 128, 256` qilib qo‘yildi. Active
budget barcha holatda `K=8`, internal steps `3`, execution fixed (`adaptive=False`).

Bu training natijasi emas. Router 32 pool bilan o‘qitilgan, kattaroq poollar
faqat mavjud tree leaf atrofidagi offset oynani kengaytiradi. Shuning uchun
screen oddiy “oyna torligi” gipotezasini tekshiradi; yangi pool bilan qayta
training qilinsa chiqadigan natijani va learned retrieval lossini tekshirmaydi.

## Natijalar

Jadvaldagi qiymatlar 32-poolga nisbatan o‘zgarish: `ΔCE` manfiy bo‘lsa yaxshi,
`Δaccuracy` esa percentage pointda berilgan.

| Model | Seed | 32-pool CE / acc | 64: ΔCE / Δacc | 128: ΔCE / Δacc | 256: ΔCE / Δacc |
|---|---:|---:|---:|---:|---:|
| 100M | 17 | 0.406425 / 85.885% | +0.003420 / −0.313 pp | +0.002861 / −0.156 pp | +0.000422 / −0.104 pp |
| 100M | 18 | 0.437259 / 83.802% | +0.001560 / +0.000 pp | −0.000588 / +0.000 pp | +0.001316 / −0.052 pp |
| 300M | 17 | 0.390924 / 86.458% | −0.000259 / −0.052 pp | +0.001274 / −0.052 pp | +0.000992 / +0.104 pp |
| 300M | 18 | 0.454808 / 83.750% | +0.000493 / −0.052 pp | −0.002614 / +0.156 pp | −0.002080 / −0.104 pp |
| 500M | 17 | 0.416870 / 86.458% | −0.001587 / −0.313 pp | −0.003124 / −0.104 pp | −0.005133 / −0.104 pp |
| 500M | 18 | 0.442592 / 84.115% | +0.001705 / +0.156 pp | +0.004027 / +0.104 pp | +0.004018 / +0.104 pp |

Olti checkpoint bo‘yicha o‘rtacha delta:

| Pool | ΔCE | Δaccuracy |
|---:|---:|---:|
| 64 | +0.000889 | −0.095 pp |
| 128 | +0.000306 | −0.009 pp |
| 256 | −0.000077 | −0.026 pp |

## Qaror

**Candidate windowni shunchaki 32 dan kattalashtirish yechim sifatida rad
qilindi.** 500M seed17’da `256` pool CE’ni `0.00513` yaxshilagan bo‘lsa ham,
seed18’da ayni o‘zgarish CE’ni `0.00402` yomonlashtirdi; 100M va 300M’da ham
yo‘nalish seedga bog‘liq. O‘rtacha ta’sir nolga yaqin va hard accuracy uchun
ijobiy, barqaror signal yo‘q.

Bu P-001 butunlay yopildi degani emas. Test faqat tree tomonidan berilgan
lokal oynaning kengligini kengaytirdi; learned candidate recall, final
corrected-output regret va retrieval loss bilan o‘qitish hali boshqa gipoteza.
Ammo mavjud dalillar bilan “32 pool torligi katta muammoning o‘zi” deyishga
asos yo‘q. Keyingi asosiy yo‘nalish selectorning target-alignmenti,
circuit specialization va recurrent state/output interface bo‘lishi kerak.

## Qayta ishlatish

```powershell
python -u benchmark_candidate_pool_screen.py `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s18_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s18_to_full_10000.pt `
  --checkpoint results/checkpoints/ne500_staged_b128_s17_full_10000.pt `
  --checkpoint results/checkpoints/ne500_staged_b128_s18_full_10000.pt `
  --candidate-pools 32 64 128 256 --batches 4 --examples-per-task 32 `
  --device cuda `
  --output results/runs/candidate_pool_screen_ne100_ne300_ne500_1920.json
```

Raw JSON ignored run artifact sifatida `results/runs/` ichida saqlanadi.
