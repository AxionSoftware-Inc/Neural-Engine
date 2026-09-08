# P-001 — Route-weight ablation audit

Sana: 2026-09-08. Maqsad: candidate selection to‘g‘ri bo‘lsa ham, selected
`K=8` circuitlarga beriladigan query-key softmax weight’lari final outputni
buzayotgan-buzmayotganini ajratish.

## Protocol

100M, 300M va 500M staged 10k checkpointlarning seed17/18 variantlari bir xil
held-out evaluator bilan tekshirildi. Har checkpointda 15 task × 32 misol × 4
batch = `1,920` misol, `K=8`, 3 internal step va `adaptive=False` ishlatildi.
Checkpoint, router path, selected circuit ID’lari va route gain’lari muzlatildi;
faqat allaqachon tanlangan circuitlar orasidagi weight almashtirildi.

Variantlar:

- `uniform`: har bir selected circuitga `1/8`;
- `softmax_power_half`: natural weight’larni yumshatish;
- `softmax_power2`: natural weight’larni keskinlashtirish;
- `top1`: faqat eng katta natural weight slotiga massa berish.

Bu retrieval testi emas: candidate poolga kirmagan circuitlar bu screen’da
ko‘rilmaydi. Shuning uchun natija P-001 retrieval headroomini o‘lchamaydi.

## Natijalar

Jadvaldagi delta natural route’ga nisbatan; CE’da manfiy yaxshi, accuracy
percentage pointda berilgan.

| Variant | O‘rtacha ΔCE | O‘rtacha Δaccuracy | 6 run’da CE yo‘nalishi |
|---|---:|---:|---|
| Uniform | `−0.000201` | `−0.009 pp` | 6/6 yaxshiroq |
| Softmax power 1/2 | `−0.000108` | `−0.009 pp` | 6/6 yaxshiroq |
| Softmax power 2 | `+0.000273` | `−0.026 pp` | 0/6 yaxshiroq |
| Top-1 | `+0.021070` | `−0.391 pp` | 0/6 yaxshiroq |

Uniform variantdagi CE foydasi barcha model/seedlarda takrorlandi, ammo hard
accuracy uchun foyda bermadi. Top-1 tajribasi selected circuitlar orasidagi
soft mixing muhimligini ko‘rsatdi: weight’ni bitta circuitga yig‘ish sifatni
sezilarli yomonlashtirdi. Shunga qaramay uniform va power-half foydasi P-001
uchun qabul qilish gate’idan juda kichik va route ID’larini o‘zgartirmaydi.

## Qaror

Route weighting kichik calibration nomutanosibligiga ega, lekin capacity
saturationning asosiy sababi emas. Uniform/flattened weight inference patchi
defaultga kiritilmadi va yangi training run boshlanmadi. P-001 ochiq qoladi:
asosiy qolgan imkoniyat candidate poolga kirmayotgan useful circuitlarni
arzon, target-aligned usulda topishdir. P-002/P-003 uchun esa bank
specialization va scaling diagnostikasi hali zarur.

**Decision:** `REJECTED AS A PRIMARY FIX`.

## Qayta ishlatish

```powershell
python -u benchmark_route_weight_ablation.py `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s18_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s18_to_full_10000.pt `
  --checkpoint results/checkpoints/ne500_staged_b128_s17_full_10000.pt `
  --checkpoint results/checkpoints/ne500_staged_b128_s18_full_10000.pt `
  --batches 4 --examples-per-task 32 --device cuda `
  --output results/runs/route_weight_ablation_ne100_ne300_ne500_1920.json
```

Raw JSON ignored run artifact sifatida `results/runs/` ichida saqlanadi.
