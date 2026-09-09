# P-007 — State-path diagnostic and extended correction-scale audit

**Sana:** 2026-09-09  
**Status:** `DIAGNOSTIC COMPLETE — NO UNIVERSAL SCALE FIX`  
**Branch:** `exp/track-runtime`

## Savol

P-007 faqat router candidate retrieval muammosi bo‘lishi shart emas. Ushbu
sinov ikki narsani ajratadi:

1. tanlangan circuit outputi recurrent state yo‘liga haqiqatan ta’sir qilyaptimi;
2. ta’sir mavjud bo‘lsa, correction amplitudasini o‘zgartirish sifatni barqaror
   yaxshilaydimi.

Bu ish checkpointlarni qayta o‘qitmaydi va default modelni o‘zgartirmaydi.

## 1. State-path diagnostikasi

100M/300M/500M staged full-bank checkpointlari seed17da, fixed execution,
15 task × 16 misol, jami 240 misol bilan tekshirildi. Circuit bankiga hook
qo‘yilib, har bir recurrent stepdagi correction delta o‘lchandi. So‘ng
`circuit_delta_scale=0` qilib correction hissasi o‘chirildi va tabiiy route
100% global/within-task cyclic route replay bilan solishtirildi.

| Model | Δ norm / encoded | Correction o‘chirilganda CE | Accuracy delta | Global replay CE delta | Within-task replay CE delta |
|---|---:|---:|---:|---:|---:|
| 100M s17 | 4.68% | −0.00865 | 0.00 pp | −0.00737 | −0.00143 |
| 300M s17 | 4.82% | −0.00472 | −0.42 pp | −0.00421 | −0.00146 |
| 500M s17 | 4.98% | −0.00622 | −0.42 pp | −0.00510 | −0.00345 |

`candidate_recall=1.0` bu yerda kuchli routing sifati dalili emas: selected
IDlar candidate pool ichidan olinadi, shuning uchun bu diagnostikadagi recall
ta’rifan trivial. Muhim signal — route almashtirilganda correction delta
o‘zgaradi (`mean change / natural delta` taxminan `1.19–1.30`). Demak circuit
outputi state yo‘liga butunlay ulanmagan emas; ammo ayrim checkpointlarda
correctionni o‘chirish CE’ni yaxshilaydi. Muammo “router umuman ishlamayapti”
emas, balki correction yo‘nalishi final task loss bilan ishonchli
moslashmagan.

## 2. Kengaytirilgan scale sweep

100M/300M/500M, seed17/18 — jami 6 checkpoint; fixed checkpoint, 15 task ×
32 misol, jami 480 misol/checkpoint. Har bir checkpointda natural execution,
global route replay va within-task route replay uchun
`scale={0, 0.1, 0.25, 0.5, 1.0}` tekshirildi. Primary diagnostik signal CE;
480 misolda hard accuracy faqat `0.2083 pp` qadamlar bilan o‘zgaradi.

| Correction scale | Mean natural CE | Δ CE vs scale 0 | Mean natural accuracy | Mean global replay CE delta | Mean within-task replay CE delta |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.428571 | 0.000000 | 84.792% | 0.000000 | 0.000000 |
| 0.10 | 0.428466 | −0.000105 | 84.757% | +0.000170 | +0.000052 |
| 0.25 | 0.428340 | −0.000231 | 84.688% | +0.000437 | +0.000198 |
| 0.50 | 0.428509 | −0.000062 | 84.792% | +0.000483 | +0.000436 |
| 1.00 | 0.430237 | +0.001665 | 84.688% | −0.000628 | −0.000256 |

Scale `0.25` aggregate CE bo‘yicha eng yaxshi ko‘rinsa ham, bu faqat
`−0.00023` va seed/model bo‘yicha bir xil emas. Har checkpointning CE-optimal
scale’i mos ravishda `100M s17=0`, `100M s18=1`, `300M s17=1`, `300M s18=0`,
`500M s17=0.25`, `500M s18=0.5` bo‘ldi. Bu universal scale yo‘qligini
ko‘rsatadi. Scale `1.0` o‘rtacha CE bo‘yicha scale `0`dan yomonroq, lekin
replay ta’sirining ishorasi ham seedlar orasida almashadi.

## Qaror

1. `circuit_delta_scale`ni 0.25 yoki boshqa bitta qiymatga default qilish
   **rad etildi**. Kichik aggregate CE farqi katta, takrorlanuvchi quality
   yutug‘i emas.
2. Correction pathni olib tashlash ham **rad etildi**: 300M/500Mda hard
   accuracy `−0.42 pp` bo‘ldi. Demak circuit bank befoyda emas; uning ta’siri
   bor, lekin ishonchli foydaga aylantirilmagan.
3. P-007ning eng aniq hozirgi ta’rifi: **router signalini kuchaytirish emas,
   selected correctionni final lossga moslaydigan state/output interface va
   training objective yetarlicha credit bermayapti**.
4. P-003 capacity-only scalingni davom ettirishga bu natija asos bermaydi.
   Keyingi tajriba yangi bank yoki scale sweep emas, correctionni final task
   loss bilan bog‘laydigan opt-in training objective bo‘lishi kerak.

## Keyingi xavfsiz tajriba

Frozen expert/circuit bank ustida kichik 100M two-seed pilot:

- router va body defaultini o‘zgartirmaslik;
- route tanlangandan keyin final logitsga ta’sir qiladigan correction uchun
  detached teacher-target emas, final CE gradientini alohida kuzatish;
- correction contributionni bounded residual bilan cheklash;
- natural hard accuracy/CE, route replay CE, correction-to-encoded ratio va
  active paramsni birga o‘lchash;
- kamida `+1 pp` held-out yoki aniq, ikki seedli CE foydasi bo‘lmasa defaultga
  qabul qilmaslik.

Bu hali implementatsiya emas; mavjud scale sweepdan kelib chiqqan keyingi
eksperiment chegarasidir.

## Reproduction

State-path diagnostic:

```powershell
python -u diagnose_p007_state_path.py `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne500_staged_b128_s17_full_10000.pt `
  --device cuda --examples-per-task 16 `
  --output results/runs/p007_state_path_scale_s17.json
```

Extended scale sweep:

```powershell
python -u benchmark_p007_correction_gain_sweep.py `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s18_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s18_to_full_10000.pt `
  --checkpoint results/checkpoints/ne500_staged_b128_s17_full_10000.pt `
  --checkpoint results/checkpoints/ne500_staged_b128_s18_full_10000.pt `
  --device cuda --examples-per-task 32 --scales 0 0.1 0.25 0.5 1.0 `
  --output results/runs/p007_correction_gain_sweep_extended_s17_s18.json
```

Raw JSONlar `results/runs/` ichida saqlanadi va generated run artefact sifatida
gitga majburan qo‘shilmaydi.
