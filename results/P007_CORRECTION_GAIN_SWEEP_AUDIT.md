# P-007 — Inference-only correction-gain sweep

**Sana:** 2026-09-07  
**Status:** `REJECTED AS A QUALITY FIX`  
**Checkpointlar:** 100M staged full-bank, seed17/18

## Savol

Circuit correction’ining final state/logitga ta’siri kichik ko‘ringani uchun
inference-time `circuit_delta_scale`ni oshirish route signalini qayta jonlantira
oladimi? Bu test training qilmaydi: checkpoint, router, circuit bank va batch
bir xil qoladi; faqat correction multiplier `0.5, 1.0, 2.0, 4.0` qilib o‘zgartiriladi.
Har scale’da natural route va cyclic global/within-task route replay o‘lchanadi.

## Natijalar

| Seed | Scale | Natural acc | Natural CE | Global replay Δacc | Global ΔCE | Within-task replay Δacc | Within-task ΔCE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.5 | 85.42% | 0.4168 | −0.42 pp | +0.0010 | −0.42 pp | −0.0004 |
| 17 | 1.0 | 85.21% | 0.4193 | −0.63 pp | −0.0003 | −0.42 pp | −0.0018 |
| 17 | 2.0 | 85.21% | 0.4232 | −0.42 pp | −0.0007 | −0.63 pp | −0.0027 |
| 17 | 4.0 | 85.21% | 0.4560 | −0.21 pp | −0.0194 | −0.42 pp | −0.0208 |
| 18 | 0.5 | 84.17% | 0.4315 | +0.42 pp | +0.0032 | +0.21 pp | +0.0020 |
| 18 | 1.0 | 84.17% | 0.4310 | +0.63 pp | +0.0046 | +0.21 pp | +0.0044 |
| 18 | 2.0 | 84.38% | 0.4343 | 0.00 pp | +0.0052 | +0.21 pp | +0.0055 |
| 18 | 4.0 | 82.29% | 0.4597 | +1.46 pp | −0.0010 | +1.04 pp | −0.0046 |

## Qaror va talqin

`scale=4` seed18 natural accuracy’ni `1.0`ga nisbatan `−1.88 pp` tushirdi va
seed17’da CE keskin yomonlashdi. Route replay natijalari seedlar o‘rtasida
qarama-qarshi: seed17da almashtirilgan route ko‘pincha yomon, seed18da esa
ba’zan yaxshiroq chiqdi. Bu correction amplitudasi oshishi barqaror causal
signalga aylanganini ko‘rsatmaydi. Accuracy farqlari 480 misolda kichik qadamli
bo‘lgani uchun CE replay asosiy diagnostik signal sifatida olindi.

Shuning uchun `circuit_delta_scale`ni 2 yoki 4 ga oshirish training retsepti yoki
default sifatida qabul qilinmadi. P-007 ochiq qoladi: keyingi ish correction
amplitudasini emas, circuit output → recurrent state interface’ini va uning
training targetini sababiyroq qilishga qaratiladi.

## Reproduction

```powershell
python -u benchmark_p007_correction_gain_sweep.py `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s18_to_full_10000.pt `
  --device cuda --examples-per-task 32 --scales 0.5 1.0 2.0 4.0 `
  --output results/runs/p007_correction_gain_sweep_ne100_s17_s18.json
```

