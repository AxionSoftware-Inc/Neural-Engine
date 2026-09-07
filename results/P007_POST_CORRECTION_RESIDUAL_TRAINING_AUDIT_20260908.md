# P-007 — Trained post-GRU correction residual audit

**Sana:** 2026-09-08  
**Branch:** `exp/track-native-engine`  
**Status:** `REJECTED FOR ADOPTION`

## Gipoteza

Oldingi inference-only sinovda `post_correction_residual_scale=1.0` ayrim
seedda route signalini kuchaytirgan, lekin seedlar qarama-qarshi chiqqan.
Ushbu nazorat ayni variantni training vaqtida ham ishlatib, inference-time
moslashuv yo‘qligi muammoni hal qiladimi degan savolni tekshiradi.

## Protocol

- Native NE-V0.12, 20M coverage-matched 5k checkpointdan continuation;
- seed 17/18 uchun control va treatment bir xil boshlang‘ich checkpoint;
- balanced train, batch `128`, CUDA/RTX 3060, `2000` continuation step;
- faqat `post_correction_residual_scale: 0.0 → 1.0` o‘zgardi;
- router, circuit bank, active budget va boshqa losslar o‘zgarmadi.

## Natijalar

| Seed | Control acc. | Treatment acc. | Δ accuracy | Control CE | Treatment CE | Δ CE |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | `74.635%` | `74.531%` | `−0.104 pp` | `0.778581` | `0.782174` | `+0.003593` |
| 18 | `74.141%` | `74.219%` | `+0.078 pp` | `0.796336` | `0.799181` | `+0.002845` |
| **Mean** | **`74.388%`** | **`74.375%`** | **`−0.013 pp`** | **`0.787458`** | **`0.790678`** | **`+0.003219`** |

Har ikki yo‘lda active parameter estimate `1.978M`, peak VRAM `522 MB` va
training vaqti taxminan `91 s` bo‘ldi. Treatment inference interface’ini
o‘zgartirgan bo‘lsa ham, quality foydasi bermadi.

## Qaror

**REJECTED FOR ADOPTION.** Training bilan residual bypassni moslashtirish ham
ikki seedda CE’ni yaxshilamadi va accuracy meanini oshirmadi. Bu yo‘lni
100M/300Mga scale qilmaymiz. P-007 uchun route-state bypass oilasi yopildi;
keyingi e’tibor candidate retrieval, subset regret va circuit specializationga
qaratiladi.

## Reproduksiya

```powershell
python -u train.py --config configs/ne_20_v12_coverage_matched.yaml --steps 2000 --device cuda --balanced-train --seed 17 --init-checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt --checkpoint results/checkpoints/p007_trainres_control_s17_2000.pt
python -u train.py --config configs/ne_20_v12_post_residual_train.yaml --steps 2000 --device cuda --balanced-train --seed 17 --init-checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt --checkpoint results/checkpoints/p007_trainres_treatment_s17_2000.pt
```

Seed18 uchun ayni buyruqlarda `--seed 18` va `ne20_v12_coverage_matched_seed18_5000.pt`
ishlatiladi. Run JSONlari `results/runs/p007_trainres_*_2000.json` nomlari bilan
saqlandi.

