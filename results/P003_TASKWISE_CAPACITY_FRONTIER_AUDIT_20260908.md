# P-003 — Task-wise capacity frontier audit

Sana: 2026-09-08. Maqsad: 100M, 300M va 500M Native Engine natijalarida
capacity umuman foyda bermayaptimi yoki faqat ayrim composition tasklarida
foyda chiqmayaptimi — shuni ajratish.

## Protocol

Oldin olingan bir xil staged 10k evaluator natijalari task bo‘yicha qayta
ajratildi. Har scale seed17/18 bilan o‘lchangan; tasklar 15 ta va depth-1,
depth-2, depth-3 guruhlari mos ravishda 9/3/3 taskdan iborat. Bu jadval
`eval_split=all` all-screen natijalarining reanalizidir; yangi training run
emas. Source JSONlar `results/runs/` ichida saqlangan.

## Depth frontier

| Scale | Depth-1 mean | Depth-2 mean | Depth-3 mean |
|---|---:|---:|---:|
| 100M | `99.67%` | `72.14%` | `53.26%` |
| 300M | `99.83%` | `71.68%` | `53.78%` |
| 500M | `99.61%` | `72.14%` | `53.71%` |

100M→500M oralig‘ida depth-1 allaqachon to‘yingan, depth-2 amalda tekis,
depth-3 esa atigi `0.45 pp` atrofida o‘zgargan. Shuning uchun qo‘shimcha
dormant parametrlar mavjud composition/state bottleneckni avtomatik yopmayapti.

## Task-wise average across all six runs

| Task | Mean accuracy |
|---|---:|
| add / subtract / greater_than / less_equal / xor_parity | `100.00%` |
| multiply / max3 / median3 / min3 | `99.50%` atrofida |
| reverse_sum | `95.70%` atrofida |
| lookup | `99.93%` atrofida |
| chain3 | `20.64%` |
| compose_add_mul | `60.55%` |
| compose_if | `92.12%` |
| state_machine | `8.41%` |

`chain3` va ayniqsa `state_machine` uch scale’da ham past qolmoqda. Shu bilan
birga `compose_if` va `compose_add_mul` qisman ishlayapti. Bu faqat classifier
umuman ishlamayotganini emas, ayrim multi-step state transitionlar uchun
representation/dataflow yetarli emasligini ko‘rsatadi.

## Talqin va qaror

Bu reanaliz “15 task yetarli emas, shuning uchun sifat ceiling” degan xulosani
tasdiqlamaydi: composition tasklar hali yechilmagan va ularning accuracy’si
capacity bilan o‘smagan. Kuchliroq xulosa shuki, hozirgi bottleneck raw bank
capacity emas, intermediate state’ni keyingi operationga aniq uzatish va
composition circuitlarini foydali train qilishdir.

Shuning uchun 700M/1B capacity-only run boshlanmaydi. Keyingi Native tajriba
routerni kattalashtirish emas, `chain3`/`state_machine` kabi depth-2/3 tasklarda
intermediate state va circuit transition signalini alohida o‘lchaydigan minimal
opt-in diagnostic bo‘ladi. Bunda depth-1 accuracy, active budget va routing
metrikalari alohida saqlanadi.

**Decision:** capacity-only scaling `REJECTED AS NEXT STEP`; composition/state
frontier P-003/P-004 uchun asosiy yo‘nalish.

## Source artifacts

- `results/runs/ne100_growth_from_ne20_s17_to_full_10000.json`
- `results/runs/ne100_growth_from_ne20_s18_to_full_10000.json`
- `results/runs/ne300_growth_b128_from_ne20_s17_to_full_10000.json`
- `results/runs/ne300_growth_b128_from_ne20_s18_to_full_10000.json`
- `results/runs/ne500_staged_b128_s17_full_10000.json`
- `results/runs/ne500_staged_b128_s18_full_10000.json`
