# P-003 typed-register 700M unseen-range audit

Status: `REJECTED AS A CAPACITY-ONLY GENERALIZATION FIX`

## Savol va protocol

700M factorized typed-register modeli seed17 500M unseen-range parent-growth
checkpointidan kengaytirildi. Model faqat operand qiymatlari `0–31`da 10,000
qadam o‘qidi; `32–63` qiymatlari trainingda berilmadi. Arxitektura supported
700M control bilan bir xil: 55,000 virtual route rows, 235 factor rows,
rank-16, active circuits `8`, internal steps `3`, batch `128`, `lazy_adamw`.

Qat’iy evaluation har birining `32^3 × 9 = 294,912` misolidan iborat bo‘ldi.

## Natija

| Model | Train range `0–31` | Unseen range `32–63` |
|---|---:|---:|
| 300M global factorized | 99.74% | 27.35% |
| 500M parent-growth | 99.85% | **28.57%** |
| 700M parent-growth, seed17 | **99.84%** | **27.74%** |

700M 500Mga nisbatan unseen range’da `−0.83 pp`, 300Mga nisbatan esa faqat
`+0.39 pp` berdi. In-range full-grid sifati `99.84%`, shuning uchun muammo
underfitting emas: model ko‘rilgan tokenlarda funksiyani o‘rganadi, ammo
ko‘rilmagan token regioniga ko‘chira olmaydi.

Per-pair unseen accuracy `17.39%–49.34%` oralig‘ida bo‘ldi; demak ayrim
operation pairlar yaxshi ko‘rinsa ham umumiy range generalization yechilmagan.

## Xarajat

- Total physical parameters: `25,785,153`.
- Estimated active path: `1,792,305` (`6.95%`).
- Training: `965.76 s` (~16.1 min) on NVIDIA GeForce RTX 3060.
- No attention or Transformer block; register graph unchanged.

## Qaror

700M capacity o‘zi unseen-value bottleneckni yechmaydi. Qo‘shimcha bank hajmi
700Mga ko‘tarilganda ham natija 500Mdan past bo‘ldi. Shuning uchun shu
protocolda 1B yoki undan katta bankni yana o‘qitish ilmiy jihatdan oqlanmaydi.

Keyingi foydali ish — qiymatlarni memoriyalashdan mustaqil representation:
equivariant/value-state interface yoki Qwen-derived activation-to-register
adapter/distillation. Bu experiment capacity muammosini umumiy arxitektura
imkonsizligi deb isbotlamaydi, lekin “ko‘proq parametr = unseen generalization”
gipotezasini rad etadi.

## Reproduction

```powershell
python grow_factorized_capacity.py `
  --parent-checkpoint results/checkpoints/ne_typed_register_500m_factorized_ood_range_growth_10000.pt `
  --target-config configs/ne_typed_register_700m_factorized_ood_range_growth.yaml `
  --output results/checkpoints/ne_typed_register_700m_factorized_ood_range_growth_init.pt

python train_composition.py `
  --config configs/ne_typed_register_700m_factorized_ood_range_growth.yaml `
  --steps 10000 --device cuda `
  --init-checkpoint results/checkpoints/ne_typed_register_700m_factorized_ood_range_growth_init.pt `
  --checkpoint results/checkpoints/ne_typed_register_700m_factorized_ood_range_growth_10000.pt `
  --run-id ne_typed_register_700m_factorized_ood_range_growth_10000 `
  --output results/runs --examples-per-task 64

python evaluate_composition.py `
  --checkpoint results/checkpoints/ne_typed_register_700m_factorized_ood_range_growth_10000.pt `
  --grid-size 32 --value-min 32 --value-max 63 --batch-size 1024 --device cuda
```

Train va unseen full-grid JSONlar `results/runs/` ichida saqlangan;
checkpointlar Git’dan ignore qilingan.
