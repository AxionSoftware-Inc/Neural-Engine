# P-004 — Recurrent state information probe

Sana: 2026-09-08. Bu inference-only diagnostic query state ichida oldingi
composition bosqichi haqidagi ma’lumot bor-yo‘qligini tekshiradi. U modelni
o‘zgartirmaydi va quality benchmark o‘rnini bosmaydi.

## Savol

`query_states[:, 1]` — birinchi circuitdan keyingi state; u depth-2/3
misollarda stage-0 targetni olib yurishi kerak. `query_states[:, 2]` esa
stage-1 target uchun keyingi operationga kirish state’idir. Train-splitda
kichik probe o‘qitilib, held-out operand kombinatsiyalarida tekshirildi.

Ikki probe ishlatildi:

- linear `state_dim → 64` classifier;
- `state_dim → 128 → GELU → 64` nonlinear classifier.

100M/300M/500M staged checkpointlarning seed17/18 juftlari, har birida 3,840
stage-0 va 1,536 stage-1 held-out misol, 400 probe epoch bilan tekshirildi.
Chance accuracy `1.5625%`.

## Natijalar

| Signal | Query step | Target | Mean held-out accuracy |
|---|---:|---:|---:|
| linear probe | 1 | stage-0 | `59.59%` |
| nonlinear probe | 1 | stage-0 | `67.27%` |
| direct model stage head | 0 | stage-0 | `67.60%` |
| linear probe | 2 | stage-1 | `45.14%` |
| nonlinear probe | 2 | stage-1 | `51.32%` |
| direct model stage head | 1 | stage-1 | `60.04%` |

Nonlinear probe linear probe’dan yaxshiroq, lekin direct stage output’dan
ustun emas. 100M’dan 500M’gacha nonlinear carry signal deyarli o‘smadi.

## Xulosa

Intermediate ma’lumot state’da butunlay yo‘qolmayapti: held-out decodability
chance’dan juda yuqori. Shu bilan birga signal to‘liq va scale bilan
monotonik kuchayadigan computation emas. Oddiy output head’ni nonlinear
qilishning o‘zi katta sakrash beradi degan gipoteza qo‘llab-quvvatlanmadi;
asosiy ceiling keyingi operation uchun state’dan foydali, barqaror algebraic
composition olishda qolmoqda.

Keyingi history/tape sinovi shu sabab tanlandi: u oldingi state’ni keyingi
query’ga skip sifatida beradi va circuit/router body’ni o‘zgartirmaydi.

## Reproduction

```powershell
python -u diagnose_state_information.py `
  --checkpoint results/checkpoints/ne100_gate_staged_s17_full_10000.pt `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s18_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s18_to_full_10000.pt `
  --checkpoint results/checkpoints/ne500_staged_b128_s17_full_10000.pt `
  --checkpoint results/checkpoints/ne500_staged_b128_s18_full_10000.pt `
  --batches 4 --examples-per-task 64 --epochs 400 --lr 0.02 --device cuda `
  --output results/runs/state_information_probe_scale_seed17_seed18_nonlinear.json
```
