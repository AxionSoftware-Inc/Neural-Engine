# P-001 — Full-bank key route counterfactual screen

Sana: 2026-09-07. Maqsad: hierarchical candidate retrieval 32 ta circuit bilan
cheklanayotgani P-001ning asosiy sababi bo‘lsa, mavjud `router.keys` bilan full
bankdan top-8 tanlash tabiiy route’ni yaxshilashi kerakligini tekshirish.

## Protocol

100M, 300M va 500M staged 10k checkpointlar seed17/18da bir xil evaluator bilan
tekshirildi: har taskdan 64 misol, 15 task, fixed execution. Avval tabiiy
hierarchical route query-state’lari olindi. So‘ng shu query-state’larning o‘zida
butun reachable bank `router.keys` bo‘yicha top-8 circuit tanlandi va barcha
internal step’lar forced route bilan qayta ishlatildi. Bu one-shot counterfactual
screen; yangi model train qilinmadi va keyingi query-state’lar qayta hisoblanmadi.

## Natijalar

| Model | Seed | Natural acc | Full-key acc | Δ accuracy | Δ CE |
|---|---:|---:|---:|---:|---:|
| 100M | 17 | 85.10% | 85.21% | +0.10 pp | +0.0018 |
| 100M | 18 | 83.85% | 83.65% | −0.21 pp | +0.0071 |
| 300M | 17 | 85.52% | 85.31% | −0.21 pp | +0.0109 |
| 300M | 18 | 83.65% | 83.33% | −0.31 pp | +0.0082 |
| 500M | 17 | 84.69% | 84.38% | −0.31 pp | +0.0152 |
| 500M | 18 | 83.54% | 83.75% | +0.21 pp | +0.0029 |

Accuracy mean delta 100M/300M/500M uchun mos ravishda `−0.05/−0.26/−0.05
pp`; CE esa barcha olti run’da yomonlashdi. 16 misol/task exploratory screen’da
ham 300M va 500M uchun ijobiy result chiqmadi; 64 misol/task repeat shu xulosani
barqarorlashtirdi.

## Qaror

**Full-bank key route P-001 yechimi sifatida rad qilindi.** Hierarchical
candidate poolni butun bank scoring bilan almashtirish o‘sha query-state’da
foydali route bermadi. Bu P-001ni butunlay yopmaydi — learned candidate recall
va final corrected-output regret hali alohida masala — lekin full-bank key
score’ni qo‘shishning o‘zi sifat sakrashi bermasligini ko‘rsatadi.

P-007 bilan birga o‘qilganda, muammo faqat circuitni topishda emas: route
almashtirilganda final output ham kuchli o‘zgarmayapti. Keyingi experiment
candidate retrievalni kattalashtirish emas, circuit correction/state interface
va cascade’ni sababiyroq qilish tomonga yo‘naladi.
