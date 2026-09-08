# P-004 — Differentiable typed intermediate-register bridge

Sana: 2026-09-08. Maqsad: Native Engine recurrent bosqichlari orasida
oraliq qiymatni faqat umumiy hidden state orqali emas, 64-class typed value
register orqali uzatish final composition quality’ni yaxshilaydimi?

## Arxitektura

Opt-in `typed_register_bridge` har bir bosqichdan keyingi `step_logits`ni
`softmax` qilib, 64 ta qiymat embedding’ining aralashmasiga aylantiradi.
Keyingi bosqich query’siga shu typed value context qo‘shiladi. Bridge basis’i
zero-init qilinadi, shuning uchun checkpoint migratsiyasidagi birinchi forward
default Native yo‘li bilan aynan bir xil. Bu attention emas; qo‘shimcha yo‘l
64 × `state_dim` value basisidan iborat.

Ikki mode tekshirildi:

- `soft`: probability-weighted value embedding;
- `straight_through`: forward’da argmax bitta value, backward’da soft gradient.

## Protocol

20M `coverage_matched_5000` seed17/18 checkpointlaridan matched 2,000-step
continuation qilindi. Har mode uchun 2×2 arms ishlatildi:

1. old model / final loss (`control`);
2. old model / final + depth-2/3 stage loss (`stage_only`);
3. typed bridge / final loss (`bridge_only`);
4. typed bridge / final + depth-2/3 stage loss (`bridge_stage`).

Har arm bir xil batch oqimi, hard routing, optimizer va `1,920` misollik
held-out evaluatorni oldi. Qabul qilish gate’i ikki seedda kamida `+2 pp`
final hard accuracy va CE regressiyasiz natija edi.

## Soft bridge natijasi

| Arm | Seed17 Δ accuracy | Seed18 Δ accuracy | Mean Δ accuracy | Mean Δ CE |
|---|---:|---:|---:|---:|
| stage_only | `+0.677 pp` | `−0.208 pp` | `+0.234 pp` | `−0.006793` |
| bridge_only | `+0.208 pp` | `−0.990 pp` | `−0.391 pp` | `−0.011325` |
| bridge_stage | `+0.990 pp` | `−0.313 pp` | `+0.339 pp` | `−0.006420` |

`bridge_stage` stage-0 accuracyni `+7.500/+6.354 pp` (`+6.927 pp` mean)
ko‘tardi, lekin stage-1 delta `+1.042/−1.693 pp`, stage-2 delta
`+2.604/−1.302 pp` bo‘ldi. Demak intermediate signal kuchaydi, ammo keyingi
bosqichlarda barqaror computation sifatida ishlamadi. `bridge_only` CE’da
yaxshilangan bo‘lsa ham hard accuracy ikki seedda bir xil emas.

## Straight-through natijasi

| Arm | Seed17 Δ accuracy | Seed18 Δ accuracy | Mean Δ accuracy | Mean Δ CE |
|---|---:|---:|---:|---:|
| stage_only | `−0.313 pp` | `−0.573 pp` | `−0.443 pp` | `+0.011092` |
| bridge_only | `+0.104 pp` | `−0.938 pp` | `−0.417 pp` | `+0.010763` |
| bridge_stage | `+0.208 pp` | `−0.156 pp` | `+0.026 pp` | `+0.008959` |

Straight-through mode ikki seedda ham CE’ni yomonlashtirdi va final accuracy
gate’iga yaqinlashmadi. Qattiq forward register hozirgi training budgetida
foydali signalni barqaror uzata olmadi.

## Qaror

Typed register bridge’ning soft va straight-through variantlari
`REJECTED FOR ADOPTION`. Bu sinov muammo faqat routerda emasligini yana bir
bor ko‘rsatdi: stage targetni yaxshilash va uni keyingi query’ga uzatishning
o‘zi final composition uchun yetarli emas. Default model o‘zgarmadi, bridge
kodda faqat opt-in sifatida qoldi va 700M/1B ga scale qilinmaydi.

Keyingi P-004 arxitektura gipotezasi umumiy 64-class outputni qayta kiritish
emas, operation-conditioned typed transition bo‘lishi kerak: current state,
operation/step va selected circuit correction alohida typed register write
qoidasi orqali birikadi. Bu keyingi testni bridge bilan aralashtirmaslik uchun
alohida benchmark qilinadi.

## Reproduction

```powershell
python -u benchmark_typed_register_bridge.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --steps 2000 --stage-loss-weight 0.1 --bridge-mode soft `
  --eval-batches 4 --eval-examples-per-task 32 --device cuda `
  --output results/runs/typed_register_bridge_2x2_ne20_seed17_seed18.json

python -u benchmark_typed_register_bridge.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --steps 2000 --stage-loss-weight 0.1 --bridge-mode straight_through `
  --eval-batches 4 --eval-examples-per-task 32 --device cuda `
  --output results/runs/typed_register_bridge_st_2x2_ne20_seed17_seed18.json
```
