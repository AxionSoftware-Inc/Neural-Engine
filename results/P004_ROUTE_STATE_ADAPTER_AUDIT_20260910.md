# P-004 — Route-conditioned circuit state-write adapter audit

Sana: 2026-09-10. Bu tajriba router target distillation oilasi yopilgandan
keyingi Native Engine composition/state interface sinovi.

## Gipoteza

Hozir circuit correction GRU state-write ichida umumiy transform orqali
aralashadi. Har bir circuit uchun kichik low-rank state-write adapter berilsa,
route identity correctionning keyingi state’ga qanday yozilishini boshqarishi
va composition tasklaridagi circuit specialization yaxshilanishi mumkin.

Bu oddiy correction scale, bounded gate yoki post-GRU residual emas. Adapter
selected circuitlarning `delta` qiymatidan route-conditioned low-rank residual
hosil qiladi va uni GRUCell’ga kiruvchi `update`ga qo‘shadi:

```text
selected circuit delta -> circuit-specific low-rank adapter -> GRUCell -> next state
```

Adapter `up` matritsasi zero-init qilingan, shuning uchun eski checkpoint bilan
birinchi forward control bilan exact parity saqlanadi. Inference’da faqat
tanlangan `K=8` circuit adapter qatorlari ishlatiladi.

## Protokol

- Native 20M `coverage_matched_5000` checkpoint, seed17 va seed18;
- control: eski `circuit_state_adapter_rank=0` model;
- treatment: `circuit_state_adapter_rank=4`, scale `1.0`;
- control/treatment bir xil 1,000-step balanced continuation, batch `128`;
- circuit banki va routerning asosiy tuzilishi o‘zgarmadi;
- held-out o‘rniga train.py’ning bir xil 1,024-example validation screeni
  paired quality control sifatida ishlatildi;
- active params, training time va no-stats inference timing alohida o‘lchandi.

## Natijalar

| Seed | Control CE / acc | Adapter CE / acc | ΔCE | Δaccuracy | Training sec |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.823890 / 73.62% | 0.822357 / 73.23% | −0.001533 | −0.391 pp | 46.63 → 51.72 |
| 18 | 0.830116 / 73.33% | 0.835166 / 73.41% | +0.005050 | +0.078 pp | 46.21 → 48.26 |
| **Mean** | — | — | **+0.001759** | **−0.156 pp** | — |

Composition-depth accuracy ham barqaror yaxshilanmadi: seed17 depth-2/3
`42.19%/38.67% → 41.54%/37.37%`, seed18 esa
`44.01%/35.94% → 43.62%/36.33%` bo‘ldi.

## Active cost

Rank-4 adapter har bir circuitga `2 × 384 × 4 = 3,072` parametr qo‘shadi.
Tanlangan 8 circuit uchun active circuit params `101,376 → 125,952` bo‘ldi;
total trainable params `20.247M → 24.572M` ga chiqdi. Analitik active MAC
hisobida adapter faqat selected rows uchun qo‘shildi. RTX 3060 batch-128,
stats-free 10-iteration screenida mean latency `7.147 → 7.322 ms` (~`+2.5%`),
peak VRAM `161 → 178 MB` bo‘ldi. Bu qisqa timing screeni sifatida qaraladi,
quality natijasining o‘rnini bosmaydi.

## Qaror

**REJECTED FOR ADOPTION.** Route-conditioned state-write adapter CE’ni bir
seedda yaxshilagan bo‘lsa ham hard accuracy ikki seed meanida pasaydi va
composition-depth signal barqaror ko‘tarilmadi. Qo‘shimcha active parametr va
VRAM ham sifat foydasini oqlamadi.

Bu natija circuit identity state’ga umuman ta’sir qila olmaydi degani emas;
faqat mustaqil per-circuit low-rank write adapter hozirgi representation va
training objective bilan yetarli composition signal bermadi. Rankni 8/16 ga
oshirish yoki 300M ga ko‘chirish asoslanmaydi.

P-004 va P-002 ochiq qoladi. Keyingi Native Engine yo‘li yana per-circuit
adapter kattalashtirish emas, reusable algebraic value/state primitive yoki
composition uchun aniq typed representationni tekshirishdir.

## Reproduksiya

```powershell
python -u train.py --config configs/ne_20_v12_coverage_matched.yaml `
  --steps 1000 --device cuda --balanced-train --seed 17 `
  --init-checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/p004_route_adapter_control_s17_1000.pt `
  --run-id p004_route_adapter_control_s17_1000 --output results/runs

python -u train.py --config configs/ne_20_v12_circuit_state_adapter.yaml `
  --steps 1000 --device cuda --balanced-train --seed 17 `
  --init-checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/p004_route_adapter_treatment_s17_1000.pt `
  --run-id p004_route_adapter_treatment_s17_1000 --output results/runs
```

Seed18 uchun checkpoint va `--seed 18` almashtiriladi. Inference timing:

```powershell
python benchmark.py --checkpoint results/checkpoints/p004_route_adapter_control_s17_1000.pt `
  --device cuda --batch-size 128 --iterations 10 --balanced-batch --no-stats
python benchmark.py --checkpoint results/checkpoints/p004_route_adapter_treatment_s17_1000.pt `
  --device cuda --batch-size 128 --iterations 10 --balanced-batch --no-stats
```

Artefaktlar: `neural_engine/model.py`, `train.py`,
`configs/ne_20_v12_circuit_state_adapter.yaml`.
