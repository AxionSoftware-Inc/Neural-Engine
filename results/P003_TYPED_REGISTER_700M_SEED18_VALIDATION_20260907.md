# P-003 typed-register 700M seed-18 validation

Status: `VALIDATED CONTROL — NO ADDITIONAL CAPACITY GAIN`

## Natija

Tok uzilishidan keyin 700M factorized typed-register parent-growth yo‘li
mustaqil ikkinchi seedda qayta tiklandi. Seed-18 700M target seed-18 500M
parentdan `grow_factorized_capacity.py` orqali yaratildi va 10,000 qadam
`lazy_adamw` bilan o‘qitildi.

Bir xil qat’iy `64^3` operand gridining barcha to‘qqiz ordered operation pairi
(`2,359,296` misol) quyidagicha chiqdi:

| Model | Seed 17 | Seed 18 | Ikki seed mean |
|---|---:|---:|---:|
| 500M factorized parent-growth | 99.6569% | 99.6675% | 99.6622% |
| 700M factorized parent-growth | 99.6627% | 99.5884% | **99.6255%** |
| 700M − 500M | +0.0058 pp | −0.0791 pp | **−0.0367 pp** |

Seed-18 training reportidagi random balanced mini-eval `99.6528%` bo‘ldi;
qaror uchun asosiy raqam full-grid natijasidir.

## Xarajat va arxitektura

- Target physical parameter count: `25,785,153`.
- Estimated active path: `1,792,305` (`6.95%`), active circuit rows
  `202,784`.
- `d_model/state_dim=384`, `55,000` virtual route rows, `235` reusable
  factor rows, rank `16`, active circuits `8`, internal steps `3`.
- Execution graph: `operands -> partial -> final -> readout`; attention va
  Transformer block ishlatilmaydi.
- Seed-18 10k training vaqti: `974.23 s` (~16.2 min) RTX 3060’da.

## Texnik tuzatish

Growth initializer dastlab configuration-factory xatosi sabab ishlamadi:
`make_model()` direct-v0 uchun bo‘lgan `circuit_delta_scale` va
`correction_gate_mode` keywordlarini typed-register konstruktorga ham uzatgan.
`train.py` typed-register branchida bu ikki direct-only keyword chiqarib
tashlandi. Keyin to‘liq test suite `135 passed, 2 warnings` bilan o‘tdi va
growth initializer barcha mos tensor prefixlarini muvaffaqiyatli ko‘chirdi.

Bu tuzatish typed-register model matematikasini yoki default routingni
o‘zgartirmaydi; u faqat mavjud config/checkpoint API mosligini tiklaydi.

## Qaror

700M parent-growth yo‘li ikki seedda ham ~99.6% supported all-range modular
composition sifatini saqladi. Bu sparse non-Transformer arxitektura uchun
ijobiy quality-control natijasi va parent-growth retseptini tasdiqlaydi.

Ammo 500M dan 700M ga o‘tish ikki seed mean bo‘yicha yaxshilanish bermadi.
Shu sababli P-003 direct-v0 yo‘nalishidagi capacity scaling muammosi yopilgan
deb hisoblanmaydi va 1B ga faqat hajm uchun o‘tish hozircha rad etiladi.
Keyingi asosiy yo‘l: `0–31`da o‘qitilib, `32–63`da tekshirilganda ~29% bo‘lib
qolayotgan unseen-value generalization uchun representation yoki Qwen-derived
activation-to-register transfer. Qwen Transformer neuronlarini xom ko‘chirish
emas, moslashtiruvchi adapter/distillation alohida gate bilan tekshiriladi.

## Reproduction

```powershell
python grow_factorized_capacity.py `
  --parent-checkpoint results/checkpoints/ne_typed_register_500m_factorized_growth_seed18_10000.pt `
  --target-config configs/ne_typed_register_700m_factorized_growth_seed18.yaml `
  --output results/checkpoints/ne_typed_register_700m_factorized_growth_seed18_init.pt

python train_composition.py `
  --config configs/ne_typed_register_700m_factorized_growth_seed18.yaml `
  --steps 10000 --device cuda `
  --init-checkpoint results/checkpoints/ne_typed_register_700m_factorized_growth_seed18_init.pt `
  --checkpoint results/checkpoints/ne_typed_register_700m_factorized_growth_seed18_10000.pt `
  --run-id ne_typed_register_700m_factorized_growth_seed18_10000 `
  --output results/runs --examples-per-task 64

python evaluate_composition.py `
  --checkpoint results/checkpoints/ne_typed_register_700m_factorized_growth_seed18_10000.pt `
  --grid-size 64 --batch-size 1024 --device cuda
```

Full-grid JSONlar lokal `results/runs/` ichida saqlanadi; checkpointlar Git’dan
ignore qilingan.
