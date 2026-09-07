# P-007 — Post-GRU correction residual audit

**Sana:** 2026-09-07  
**Status:** `REJECTED FOR ADOPTION`  
**Checkpointlar:** 100M staged full-bank, seed17/18

## Gipoteza

Hozir circuit correction GRU update’iga beriladi; GRU uni qisman yutib
yuborishi mumkin. Opt-in `post_correction_residual_scale=β` correction’ning
`β × delta` qismini GRU’dan keyin state’ga bevosita qo‘shadi. Default `β=0`
bo‘lib, eski yo‘l o‘zgarmaydi. Test inference-only: checkpoint va batch
o‘zgarmadi, `circuit_delta_scale=1.0` saqlandi.

## Natijalar

| Seed | β | Natural acc | Natural CE | Global replay Δacc | Global ΔCE | Within-task replay Δacc | Within-task ΔCE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.00 | 85.21% | 0.4193 | −0.63 pp | −0.0003 | −0.42 pp | −0.0018 |
| 17 | 0.25 | 85.42% | 0.4169 | −0.42 pp | +0.0007 | −0.62 pp | +0.0008 |
| 17 | 0.50 | 85.63% | 0.4142 | +0.21 pp | +0.0028 | −0.83 pp | +0.0039 |
| 17 | 1.00 | 86.46% | 0.4128 | −0.42 pp | +0.0057 | −1.25 pp | +0.0062 |
| 18 | 0.00 | 84.17% | 0.4310 | +0.63 pp | +0.0046 | +0.21 pp | +0.0044 |
| 18 | 0.25 | 84.17% | 0.4351 | +0.21 pp | +0.0004 | +0.63 pp | +0.0005 |
| 18 | 0.50 | 84.17% | 0.4369 | 0.00 pp | +0.0011 | +0.63 pp | −0.0023 |
| 18 | 1.00 | 83.75% | 0.4436 | +0.42 pp | −0.0043 | +0.42 pp | −0.0052 |

## Qaror

Residual bypass seed17’da `β=1.0` bilan `+1.25 pp` accuracy va `−0.0065` CE
berdi, ammo seed18’da `−0.42 pp` va `+0.0126` CE regressiya bo‘ldi. Global va
within-task replay CE signali ham ikki seedda qarama-qarshi. Demak correction’ni
GRU’dan keyin majburan chiqarish route causalityni barqaror yaxshilamadi.

Bu natija retraining bilan bu interface hech qachon ishlamaydi degani emas;
faqat mavjud checkpointda inference-time bypass xavfsiz quality fix emas.
Shuning uchun variant opt-in API sifatida saqlandi, training/defaultga
ko‘chirilmaydi. P-007 ochiq qoladi; keyingi ish correction’ni qayerga qo‘shish
emas, circuit specialization va final-target credit assignmentni birgalikda
tekshirishga qaratiladi.

## Reproduction

```powershell
python -u benchmark_p007_post_residual_sweep.py `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s18_to_full_10000.pt `
  --device cuda --examples-per-task 32 --betas 0 0.25 0.5 1.0 `
  --output results/runs/p007_post_residual_sweep_ne100_s17_s18_beta1.json
```

