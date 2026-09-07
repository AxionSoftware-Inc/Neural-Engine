# P-007 — Route-final-target auxiliary loss audit

**Sana:** 2026-09-08  
**Branch:** `exp/track-native-engine`  
**Qaror:** `NOT PROMOTED` — opt-in tadqiqot patchi sifatida qoldirildi.

## Gipoteza

Route-causal diagnostic circuit delta'sining recurrent update va input
reinjection ichida kichik signalga aylanayotganini ko'rsatdi. Ushbu sinovda
inference path o'zgartirilmaydi: har bir bajarilgan stage'dagi tanlangan
circuit delta alohida `model.output(delta)` orqali yakuniy targetga qarshi
auxiliary CE oladi. Maqsad — selected circuit correctionni bevosita task
signaliga aylantirish va keyin recurrent state uni yo'qotib yuborishini
kamaytirish.

Patch faqat `route_final_target_weight > 0` bo'lganda ishlaydi. Active circuit
soni, router, circuit bank, model output va inference hisoblash yo'li
o'zgarmaydi. Route delta'lar diagnostika/training uchun stats'da qaytariladi.

## Nazoratli protokol

- Native NE-V0.12, 20M, `ne_20_v12_coverage_matched`.
- Seedlar: `17`, `18`.
- Har bir treatment va control bir xil 5k-step checkpointdan continuation.
- Balanced training, batch `128`, CUDA/RTX 3060, bir xil evaluator.
- `1000` continuation: weight `0.05` va `0.25` sweep.
- `2000` continuation: weight `0.10` va matched control.
- Metrikalar: held-out exact accuracy, validation CE, dead circuit fraction.

## Natijalar

| Continuation | Seed | Control acc. | Treatment acc. | Delta | Control CE | Treatment CE |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 17 | 73.776% | 74.505% (`w=.05`) | +0.729 pp | 0.815755 | 0.813821 |
| 1000 | 18 | 73.438% | 73.229% (`w=.05`) | −0.208 pp | 0.840512 | 0.828592 |
| 1000 | 17 | 73.776% | 73.776% (`w=.25`) | +0.000 pp | 0.815755 | 0.822877 |
| 1000 | 18 | 73.438% | 73.047% (`w=.25`) | −0.391 pp | 0.840512 | 0.827658 |
| 2000 | 17 | 74.271% | 74.766% (`w=.10`) | +0.495 pp | 0.790717 | 0.774552 |
| 2000 | 18 | 74.818% | 75.078% (`w=.10`) | +0.260 pp | 0.790904 | 0.792677 |

The 0.10 two-seed mean accuracy delta is **+0.378 pp**. The 0.05 sweep
averages **+0.260 pp**, while the 0.25 sweep averages **−0.195 pp**. The
positive 0.10 signal is therefore weak relative to the project adoption gate
and is not yet evidence that the route/state interface problem is solved.

Active estimate stayed at about `1.98M` in every run. Dead-circuit fractions
remained low but varied slightly: the 0.10 treatment was `0.284%/0.639%`
versus controls `0.213%/0.284%` for seeds 17/18. This patch did not buy a
capacity or active-path change.

## Qaror va talqin

1. Auxiliary loss selected circuit delta'ni training signaliga bog'lay oladi;
   0.10 weight'da ikkala seed ham kichik positive movement berdi.
2. Improvement `+2 pp` quality gate'iga yetmadi va weight sweep seed-stable
   monotonic trend bermadi.
3. Shuning uchun patch **defaultga kiritilmadi**, 100M/300M ga scale
   qilinmadi va P-007 `ACTIVE` bo'lib qoladi.
4. Bu natija route muammosi faqat lossda ekanini ko'rsatmaydi. Keyingi asosiy
   yo'nalish route/state interface'ini va runtime dispatchni alohida tekshirish
   bo'ladi; yangi Native losslar ketma-ket ko'r-ko'rona scale qilinmaydi.

## Reproduksiya artefaktlari

- Kod: `neural_engine/model.py`, `train.py`.
- Configlar: `configs/ne_20_v12_route_target_005.yaml`,
  `configs/ne_20_v12_route_target_01.yaml`,
  `configs/ne_20_v12_route_target_025.yaml`.
- JSON runlar: `results/runs/p007_*_ne20_*.json`.
- Unit test: `tests/test_forward.py`; to'liq suite patchdan keyin `148 passed`.
