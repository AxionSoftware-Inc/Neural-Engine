# P-007 — Input reinjection schedule audit

Sana: 2026-09-07. Maqsad: encoded input har internal step’da to‘liq qayta
qo‘shilgani circuit correctionni bosib ketayotgan bo‘lsa, birinchi step’da
to‘liq, keyingi step’larda kamayuvchi reinjection route ta’sirini oshiradimi?

## Protocol

100M/300M/500M staged 10k checkpointlar inference-only tekshirildi. Har taskdan
64 misol, 15 task, fixed execution. Checkpointlar bir xil qolgan; faqat model
update’ida schedule almashtirilgan:

| Schedule | Ma’nosi |
|---|---|
| `[1, 1, 1]` | amaldagi control |
| `[1, 0.75, 0.5]` | yumshoq kamayish |
| `[1, 0.5, 0.25]` | kuchliroq kamayish |
| `[1, 0.25, 0]` | keyingi step’larda deyarli faqat state/circuit |
| `[1, 0, 0]` | input faqat birinchi step’da |

## Natijalar

64 misol/task, seed17/18 bo‘yicha yumshoq `[1,0.75,0.5]` schedule natijasi:

| Model | Seed | Control acc | Schedule acc | Δ acc | Control CE | Schedule CE | Δ CE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100M | 17 | 85.10% | 85.21% | +0.10 pp | 0.4188 | 0.4344 | +0.0156 |
| 300M | 17 | 85.52% | 85.63% | +0.10 pp | 0.4013 | 0.4166 | +0.0153 |
| 500M | 17 | 84.69% | 84.79% | +0.10 pp | 0.4246 | 0.4379 | +0.0133 |
| 100M | 18 | 83.85% | 84.17% | +0.31 pp | 0.4474 | 0.4554 | +0.0081 |
| 300M | 18 | 83.65% | 83.65% | +0.00 pp | 0.4542 | 0.4621 | +0.0079 |
| 500M | 18 | 83.54% | 83.54% | +0.00 pp | 0.4586 | 0.4671 | +0.0084 |

Accuracy mean delta `+0.10 pp`, CE esa `+0.0114` ga yomonlashdi. Kuchliroq
schedule `[1,0.5,0.25]`, `[1,0.25,0]` va `[1,0,0]` kichik screen’da barcha
scale’larda barqaror regressiya berdi; masalan seed17da 300M accuracy mos
ravishda `−0.42/−4.17/−4.17 pp` bo‘ldi.

## Qaror

**REJECTED FOR ADOPTION; trainingga o‘tkazilmadi.** Input reinjectionni oddiy
schedule bilan kamaytirish route correctionni yetarli darajada foydali
qilmadi. Bu P-007 kuzatuvini kuchaytiradi: circuit delta kichik bo‘lsa ham,
muammo faqat input miqdori emas — state/circuit interfeysining o‘zi va
cascade composition masalasi qolmoqda.

## Reproduction artifacts

- `results/diagnostic_reinjection_schedule_scale_16.json`
- `results/diagnostic_reinjection_schedule_mild_64.json`
- Opt-in implementation: `neural_engine/model.py`, `train.py`.
