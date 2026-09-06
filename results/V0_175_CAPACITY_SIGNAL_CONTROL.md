# V0.175 — Capacity signal: controlled allocation vs learned routing

## Maqsad

Ekspert tahlilidagi asosiy savolni tekshirish: bankdagi circuitlar soni
oshganda yangi mustaqil bilim paydo bo‘ladimi va model undan foydalana oladimi?
Bu sinov Qwen transplant/router yo‘lidan alohida. Bir xil kichik Neural Engine
arxitekturasi, bir xil task/value split, bir xil `active_circuits=2` va bir xil
training budget ishlatiladi.

Ikki routing sharti bor:

- `learned`: mavjud HierarchicalRouter circuitlarni o‘zi tanlaydi;
- `controlled_task`: task ID oldindan circuit guruhiga biriktiriladi. Bu router
  tanlovini olib tashlaydi, lekin circuitlarning o‘zi mustaqil parametr qatorlari
  bo‘lib qoladi.

`controlled_task` mappingi `task_id % floor(num_circuits / active_circuits)`
orqali guruh tanlaydi. Bank kichik bo‘lsa tasklar ataylab bir guruhni bo‘lishadi;
bank kattalashganda ko‘proq alohida guruhlar paydo bo‘ladi. Bu kapasitetning
“saqlangan parametr” emas, real ishlatilgan parametr ekanini tekshiradi.

## Implementatsiya

- `train.py` ga `--routing-mode`, `--num-circuits`, `--active-circuits` va
  `--seed` override'lari qo‘shildi.
- `controlled_task_route_ids()` fixed-route nazoratini hosil qiladi.
- `configs/ne_capacity_signal.yaml` bir xil benchmark konfiguratsiyasini beradi:
  `d_model=128`, `state_dim=128`, `rank=8`, `active_circuits=2`, `candidate_pool=8`,
  `internal_steps=3`, train/eval split `train`, held-out split `heldout`.
- Unit testlar `tests/test_training_controls.py` ga qo‘shildi.

## Pilot: 1000 qadam, seed 17

| Bank | Routing | Total params | In-domain exact | Held-out exact | Dead circuits |
|---:|---|---:|---:|---:|---:|
| 8 | learned | 246,496 | 37.86% | 27.14% | 0.0% |
| 8 | controlled | 246,496 | 36.88% | 27.21% | 0.0% |
| 16 | learned | 264,928 | 40.08% | 26.33% | 0.0% |
| 16 | controlled | 264,928 | 40.34% | 27.58% | 0.0% |
| 32 | learned | 301,792 | 38.07% | 25.86% | 0.0% |
| 32 | controlled | 301,792 | 38.31% | 26.54% | 6.25% |

Natija 1000 qadamda monotonik scaling signalini bermadi. `learned` va
`controlled_task` farqi ham kichik; shuning uchun hozircha muammo faqat router
emas, optimization/training signal yoki circuitlarning foydali mustaqil
spetsializatsiyaga aylanishi bo‘lishi mumkin. 32-bankda 30/32 circuit ishladi:
15 task va har taskga 2 circuit ajratilgani sababli ikki row hali ishlatilmagan.

## Keyingi tekshiruv

## To‘liq screen: 5000 qadam, seed 17

| Bank | Routing | Total params | In-domain exact | Held-out exact | Dead circuits |
|---:|---|---:|---:|---:|---:|
| 8 | learned | 246,496 | 62.94% | 46.64% | 0.0% |
| 8 | controlled | 246,496 | 63.88% | 50.96% | 0.0% |
| 16 | learned | 264,928 | 62.79% | 48.12% | 0.0% |
| 16 | controlled | 264,928 | 64.48% | 49.82% | 0.0% |
| 32 | learned | 301,792 | 62.32% | 48.18% | 0.0% |
| 32 | controlled | 301,792 | 62.89% | 52.50% | 6.25% |

5000 qadamdan keyin ham learned routingda sig‘im oshishi sifatga aylanmadi:
held-out `46.64% → 48.12% → 48.18%` bo‘ldi, in-domain esa `62.94% →
62.79% → 62.32%` pasaydi. Controlled allocation 32-bankda eng yaxshi held-out
natijani berdi (`52.50%`), ammo 8-bankdan o‘sish atigi `+1.54` punkt. Shuning
uchun katta capacity gain hali isbotlanmadi; asosiy bottleneck routing va
specializationni foydali, generalizable circuit bilimiga aylantirishda.

Pilot va bitta full seed yetarli emas. Keyingi bosqich — ayni 6 shartni kamida
ikkinchi seed bilan takrorlash. Talqin qoidasi:

1. controlled ham learned ham oshsa — circuit capacity foydali, learned routing
   alohida bottleneck bo‘lishi mumkin;
2. controlled oshib, learned oshmasa — routing/specialization muammosi;
3. ikkalasi ham oshmasa, dense yoki training-budget control oshsa — native
   circuit bank qo‘shimcha sig‘imni foydali holatga keltira olmayapti;
4. in-domain oshib, held-out oshmasa — raw capacity bor, lekin generalization
   muammosi alohida.

Pilot commandlari:

```powershell
python train.py --config configs/ne_capacity_signal.yaml --steps 1000 --device cuda --balanced-train --num-circuits 8 --active-circuits 2 --routing-mode learned --run-id capacity_pilot_c8_learned_s17
python train.py --config configs/ne_capacity_signal.yaml --steps 1000 --device cuda --balanced-train --num-circuits 8 --active-circuits 2 --routing-mode controlled_task --run-id capacity_pilot_c8_controlled_task_s17
```

## Ikki seedli tasdiq: 5000 qadam

| Bank | Routing | Seed 17 held-out | Seed 18 held-out | Mean held-out | Mean in-domain |
|---:|---|---:|---:|---:|---:|
| 8 | learned | 46.64% | 49.01% | 47.82% | 62.74% |
| 8 | controlled | 50.96% | 48.52% | 49.74% | 63.41% |
| 16 | learned | 48.12% | 43.39% | 45.76% | 62.63% |
| 16 | controlled | 49.82% | 48.23% | 49.02% | 64.16% |
| 32 | learned | 48.18% | 46.17% | 47.18% | 62.64% |
| 32 | controlled | 52.50% | 51.22% | 51.86% | 63.58% |

Controlled allocation learned routingdan 8/16/32 banklarda mos ravishda
`+1.92`, `+3.26`, `+4.68` held-out punkt ustun bo‘ldi. Controlled 8→32 mean
held-out o‘sishi `+2.12` punkt; bu foydali, lekin katta capacity sakrashi emas.
Learned routingda 8→32 mean held-out `47.82% → 47.18%` bo‘lib, yaxshilanish
yo‘q. 32-bank controlled holatida 15 task × 2 active circuit sabab 30/32 row
ishlatildi; ikki row hali ajratilmagan.

## Qaror

Ekspert gipotezasi qo‘llab-quvvatlandi: qo‘shimcha bank rows o‘zi yetarli emas,
ularni task/domain bo‘yicha foydali mustaqil circuitlarga aylantirish va
generalizable signal bilan o‘qitish kerak. Learned global router hozircha
qo‘shimcha sig‘imni ishonchli ishlata olmayapti. Shuning uchun hozircha 300M,
700M yoki 1B ga ko‘tarilishdan oldin routing/specialization protokolini
arxitektura darajasida yaxshilash kerak.

Keyingi mantiqiy qadam — task ID'ga qattiq bog‘lanmagan, lekin controlled
allocationdagi specialization signalini saqlaydigan semantic/family router
va task/domain-balanced expert trainingini sinash. Bu bosqichda Qwen teacher
transplantini asosiy yo‘lga qaytarmaymiz: avval native circuit bank sig‘imining
foydali bilimga aylanishi isbotlanadi.
