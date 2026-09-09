# P-003 — Stable-prefix bank growth pilot

**Sana:** 2026-09-09  
**Status:** `REJECTED AS A MATERIAL QUALITY FIX; ONE-SEED PILOT`  
**Branch:** `exp/track-runtime`

## Gipoteza

Staged growthda 1408 parent circuitlari va ularning router key’lari yangi bank
bilan birga qayta o‘rganilganda, oldingi foydali route representation drift
qilishi mumkin. Yangi capacity sifati oshirmayotganining sababi shu bo‘lsa,
oldingi bank prefixini muzlatish yangi qatorlarni foydali computationga
aylantirishi kerak.

## Protocol

100M seed17 uchun bir xil `ne100_clamp_warmstart_s17_5000.pt` checkpointdan
boshlanib, full 7552-bank configda 2000 qadam train qilindi. Ikkala qo‘l bir
xil balanced batch, optimizer, learning rate va full routing state ishlatdi:

- `stable-prefix`: inherited 1408 circuit rows va 1408 router key rows
  gradient/AdamW update’dan himoyalandi; yangi rows, shared state, encoder va
  output o‘rganishda qoldi;
- `unfrozen control`: barcha parametrlar odatdagidek o‘rganildi.

Freeze implementation weight decay’ni ham hisobga oladi: eski prefix har
optimizer qadamidan keyin snapshot qiymatiga tiklanadi. Shu sabab bu faqat
gradientni nolga ko‘paytiradigan nomukammal freeze emas.

## Natijalar

| Variant | 2000-step all accuracy | Validation CE | Dead fraction | Used circuits | Training time | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Stable prefix, 1408 frozen | 80.47% | 0.580555 | 5.03% | 7,172 / 7,552 | 151.6 s | 2706 MB |
| Unfrozen control | 80.47% | 0.575009 | 5.51% | 7,136 / 7,552 | 143.9 s | 1946 MB |
| Stable − control | 0.00 pp | +0.005546 | −0.48 pp | +36 | +5.35% | +760 MB |

Hard accuracy teng bo‘lsa ham, primary CE stable-prefixda `+0.00555`
yomonroq. Circuit utilizationdagi `36` qatorlik farq sifat yutug‘iga
aylanmadi. Freeze hook/snapshot overhead ham bu variantni arzonroq qilgani
yo‘q; peak VRAM sezilarli oshdi.

## Qaror

Bu bitta seedli 2000-step screen’da stable-prefix foyda bermadi va to‘liq
seed18/5000 continuationga kengaytirilmadi. Natija “route drift umuman
muammo emas” degan qat’iy isbot emas, lekin hozirgi bottleneckni faqat eski
bankni muzlatish bilan hal qilib bo‘lmasligini ko‘rsatadi. Variant defaultga
qabul qilinmadi; opt-in training control sifatida kodda qoldi.

Keyingi yo‘l: yana bank prefixini muzlatish emas, circuit outputni final task
lossga moslaydigan specialization/representation mexanizmini tekshirish.

## Reproduction

```powershell
python -u train.py `
  --config configs/ne_100_v12_coverage.yaml `
  --init-checkpoint results/checkpoints/ne100_clamp_warmstart_s17_5000.pt `
  --steps 2000 --device cuda --balanced-train `
  --run-id ne100_stable_prefix_s17_2000 --output results/runs `
  --log-every 500 --checkpoint results/checkpoints/ne100_stable_prefix_s17_2000.pt `
  --freeze-growth-prefix 1408
```

The paired control omits `--freeze-growth-prefix` and writes
`ne100_unfrozen_growth_s17_2000`.
