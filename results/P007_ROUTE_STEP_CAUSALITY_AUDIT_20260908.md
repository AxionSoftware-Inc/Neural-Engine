# P-007 — One-step route causality audit

**Sana:** 2026-09-08  
**Branch:** `exp/track-native-engine`  
**Status:** `DIAGNOSTIC COMPLETE — BOTTLENECK SHIFTED`

## Maqsad

Oldingi route replay natijalarida final hard accuracy juda kam o‘zgargani
uchun route correction recurrent state ichida yo‘qolib ketayaptimi degan savol
ochiq qolgan edi. Bu diagnostika butun route’ni almashtirmaydi: har safar
faqat bitta internal step’ning selected circuitlari boshqa sample’dan olinadi.
Keyingi step query’si, har bir step logiti, route delta va final logit alohida
solishtiriladi.

## Protocol

- Native 100M va 300M staged full-bank checkpointlar;
- seed 17/18, held-out evaluator, 16 ta example/task (`240` sample);
- `adaptive=False`, model weights va active budget o‘zgarmaydi;
- `global` va `within_task` cyclic route swap;
- changed step uchun selected IDs, weights va route gain birgalikda almashtiriladi;
- step 0, 1, 2 alohida tekshiriladi.

## Natija

| Kuzatuv | 100M/300M diapazon |
|---|---:|
| Almashtirilgan step’dagi route-delta L2 | `1.53–1.90` |
| Keyingi query’dagi L2 (step 0/1 swap) | `0.56–0.96` |
| Final logit L2 | `0.59–1.03` |
| Final CE o‘zgarishi | `−0.00431 … +0.00211` |
| Hard accuracy o‘zgarishi | `−1.25 … +0.42 pp` |

Route o‘zgargan stepning o‘zida step-logit farqi taxminan `0.91–1.53 L2`
bo‘ldi, keyingi recurrent step’larda esa farq kamaygan bo‘lsa ham nolga
tushmadi. 100M va 300Mda bu naqsh deyarli bir xil. Demak circuit correction
state yo‘liga causal ravishda kiradi; “router tanlaydi, lekin state uni butunlay
yutib yuboradi” degan kuchli talqin tasdiqlanmadi.

Biroq cyclic swap route’ni targetga yaqinroq qilmaydi: ayrim seed/tasklarda CE
yaxshilanadi, ayrimlarida yomonlashadi, hard accuracy esa barqaror yo‘nalishga
ega emas. Ichki state sezgirligi borligi final tanlovning targetga mosligini
anglatmaydi.

## Qaror va keyingi yo‘l

- P-007ning “route interface umuman causal emas” qismi zaiflashdi, ammo P-007
  yopilmadi: route foydasi hali targetga ishonchli aylanmayapti.
- Learned bounded gate, statik correction scale, post-GRU bypass va
  route-final-target auxiliary lossni yana scale qilishga asos yo‘q.
- Asosiy bottleneck hozir `candidate retrieval / subset regret / circuit
  specialization` hamda final classifier marginiga ko‘chdi. Bu P-001/P-002
  bilan bir xil yo‘nalish.
- Keyingi quality tajribasi route-state bypass emas, counterfactual best-route
  signalini targetga moslashtiruvchi minimal retrieval/regret diagnostikasi
  bo‘lishi kerak. Bankni 700M/1Bga oshirish hozircha kerak emas.

## Reproduksiya

```powershell
python analyze_route_step_causality.py `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne100_growth_from_ne20_s18_to_full_10000.pt `
  --examples-per-task 16 --device cuda `
  --output results/runs/route_step_causality_ne100_s17_s18.json

python analyze_route_step_causality.py `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s17_to_full_10000.pt `
  --checkpoint results/checkpoints/ne300_growth_b128_from_ne20_s18_to_full_10000.pt `
  --examples-per-task 16 --device cuda `
  --output results/runs/route_step_causality_ne300_s17_s18.json
```

