# V0.178 — Frozen-bank route retrieval vs selection audit

## Maqsad

V0.177 dan keyin asosiy savol “bankda foydali circuitlar bormi yoki router ularni
topolmayaptimi?” edi. Ekspert tavsiyasi bo‘yicha 32-circuit global learned
checkpointlar muzlatildi va har bir recurrent qaror alohida counterfactual
tarzda almashtirildi:

1. oddiy learned route;
2. joriy 8-circuit candidate pool ichidagi barcha juftliklardan eng yaxshisi;
3. butun 32-circuit bank ichidagi 496 juftlikdan eng yaxshisi.

Bir qaror almashtirilganda target juftlik uchun hamma variantda bir xil
`uniform weights` va `gain=1` ishlatildi. Qolgan recurrent qadamlar modelning
odatiy learned routingi bilan davom etdi. Shunday qilib weight/gain o‘zgarishi
route ID ta’siriga aralashmadi. Bu global oracle emas, bir qarorli audit.

## Natija

Har seedda held-out splitdan 2 ta balanced batch, jami 510 misol va 3 recurrent
qaror tekshirildi. Losslar target qarorlar bo‘yicha o‘rtacha cross-entropy.

| Seed | Native learned | Fixed learned pair | Candidate oracle | Full-bank oracle | Candidate gain | Full gain | Retrieval headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 2.507 | 2.504 | 2.242 | 2.047 | 0.262 | 0.458 | 0.195 |
| 18 | 2.469 | 2.463 | 2.187 | 2.017 | 0.276 | 0.447 | 0.171 |

Bu yerda:

- `Candidate gain = fixed learned pair loss − candidate oracle loss`;
- `Full gain = fixed learned pair loss − full-bank oracle loss`;
- `Retrieval headroom = full gain − candidate gain`.

Accuracy ham xuddi shu yo‘nalishni ko‘rsatdi:

| Seed | Native | Candidate oracle | Full-bank oracle |
|---:|---:|---:|---:|
| 17 | 46.86% | 50.00% | 52.42% |
| 18 | 46.47% | 50.39% | 54.58% |

## Talqin

Bu natija sig‘im muammosini bitta sababga qisqartirmaslik kerakligini ko‘rsatdi:

1. **Candidate oracle ham ancha yaxshi.** Demak hozirgi 8-candidate ichida
   foydali juftlik mavjud, lekin learned selection uni ishonchli tanlamaydi.
2. **Full-bank oracle yana qo‘shimcha yaxshiroq.** Demak top-8 candidate retrieval
   ham alohida yo‘qotish manbai; kerakli circuitlar ko‘pincha candidate poolga
   kirmaydi.
3. Ikki seedda ham bir xil tartib saqlandi. Bu tasodifiy bitta seed effekti emas.
4. Bu hali circuitlar mukammal o‘qitilganini isbotlamaydi: oracle ham faqat bitta
   recurrent qarorni almashtiradi, barcha qarorlarni birga optimallashtirmaydi.

## Keyingi arxitektura testi

Endi 32 ta router key'ni to‘liq score qiladigan, ammo executionda faqat top-2
circuitni ishlatadigan **flat router** tekshiriladi. Bu candidate retrieval
yo‘qotishini olib tashlaydi va circuit body'ni dense ishlatmaydi.

Qaror mezoni:

- flat router 1000-step pilotda global hierarchical baseline’dan aniq ustun
  bo‘lsa, 5000-step ikki-seed screen qilinadi;
- ustunlik bo‘lmasa, keyingi muammo candidate retrieval emas, learned pair
  selection yoki circuit specialization bo‘ladi;
- flat routerning barcha 32 key score xarajati inference budgetga alohida
  yoziladi — `active_circuits=2`ning o‘zi yetarli efficiency da’vosi emas.

Checkpointlar va raw JSONlar `results/checkpoints/` va `results/runs/` ostida
lokal saqlanadi; hisobot esa qayta ishlab bo‘ladigan metrikalarni jamlaydi.
