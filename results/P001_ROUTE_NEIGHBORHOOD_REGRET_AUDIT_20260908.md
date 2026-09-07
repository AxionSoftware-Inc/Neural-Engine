# P-001 — Route-neighborhood selection/retrieval regret audit

Sana: 2026-09-08. Maqsad: native route sifati past bo‘lsa, sabab candidate
pool ichidagi selector xatosimi yoki foydali circuitlar candidate poolga
umuman kirmayaptimi — shu ikkisini final corrected output CE orqali ajratish.

## Protocol

100M, 300M va 500M staged 10k checkpointlarning seed17/18 variantlari bir xil
held-out evaluator bilan tekshirildi. Har checkpointda 15 task × 16 misol =
240 ta sample, uchta internal step alohida ko‘rildi (`720` step-sample).
Checkpoint, circuit bank, active budget `K=8` va natural router o‘zgarmadi.

Har bir step uchun natural route’dagi eng kam-weight circuit bitta circuitga
almashtirildi:

1. **Local selection probe:** barcha 32 ta candidate circuit navbatma-navbat
   sinab ko‘rildi;
2. **Full-bank retrieval probe:** full reachable bankdan router-key bo‘yicha
   top-8 circuitlar ham xuddi shu one-swap qidiruvga qo‘shildi.

Har bir variantning haqiqiy final CE’si model forward orqali o‘lchandi. Weight
alternative route uchun query–key score’dan qayta hisoblandi; boshqa step’lar
natural route bilan qoldi. Bu exhaustive `K=8` subset oracle emas, shuning
uchun natijalar faqat konservativ one-swap headroomdir. Full-bank qismi
label/targetdan foydalanadigan diagnostik probe bo‘lib, deploy qilinadigan
router emas.

## Natijalar

`Local gain` natural CE’dan candidate pool ichidagi eng yaxshi one-swap CE’ga
kamayishdir. `Retrieval gain` esa local probe’dan key top-8 full-bank probe’ga
qo‘shimcha kamayishdir. `Recall` — full-bank key top-8 a’zolarining candidate
pool ichida bo‘lish ulushi.

| Model | Seed | Natural CE | Local selection gain | Extra retrieval gain | Full-key top-8 recall |
|---|---:|---:|---:|---:|---:|
| 100M | 17 | 0.45077 | 0.00468 | 0.00773 | 0.625% |
| 100M | 18 | 0.43447 | 0.00375 | 0.00558 | 0.972% |
| 300M | 17 | 0.44228 | 0.00506 | 0.00886 | 0.243% |
| 300M | 18 | 0.44626 | 0.00489 | 0.00572 | 0.503% |
| 500M | 17 | 0.48486 | 0.00380 | **0.02199** | 0.104% |
| 500M | 18 | 0.40888 | 0.00357 | **0.01238** | 0.139% |

## Talqin

- Candidate pool ichida selectorning one-swap headroom’i barcha scale’da
  `0.0036–0.0051 CE` bo‘lib, natural top-8 tanlov mukammal emas.
- Full-bank key top-8 probe local headroomdan tashqari barcha olti run’da
  foydali target-aligned alternative topa oldi: `0.00558–0.02199 CE`.
- Bank kattalashganda retrieval opportunity ayniqsa oshdi: 500M’da full-bank
  probe local selection headroomidan taxminan `3.3–5.8x` katta.
- Full-bank key top-8 ning candidate poolga tushishi `0.10–0.97%` bo‘ldi.

Bu “full-bank key routerni productionga qo‘yamiz” degani emas. Avvalgi
full-bank key screen natural qualityni izchil oshirmagan. Sababi key top-score
target cost bilan mos emas; ushbu audit esa target CE bilan key top-8 ichidagi
one-swap opportunity’ni faqat diagnostik tarzda ko‘rsatadi. Demak bizga oddiy
pool enlargement emas, hidden state’dan final corrected-output costni
taxmin qiladigan trainable retrieval/selection signal kerak.

## Qaror

P-001 **ochiq qoladi, lekin aniqroq lokalizatsiya qilindi**:

- “candidate pool 32 bo‘lgani uchun tor” — oddiy window screen tomonidan
  tasdiqlanmadi va bu yo‘l default sifatida rad qilindi;
- “candidate/key retrieval final output costni ko‘rmayapti” — one-swap probe
  tomonidan qo‘llab-quvvatlandi, ayniqsa 500M’da;
- “selector candidate ichida ham xato qiladi” — kichik, lekin barqaror local
  headroom bilan tasdiqlandi.

Keyingi arxitektura tajribasi candidate poolni shunchaki kattalashtirish yoki
full-bank scoring emas. Avval kichik frozen-bank prototipida query + candidate
output signature’dan local final-cost surrogate o‘rgatib, uning candidate
selection regretini kamaytirishini tekshirish kerak. Circuit body va active
budget o‘zgarmasin.

## Artefaktlar

- Script: `analyze_route_neighborhood.py`.
- Raw runs: `results/runs/route_neighborhood_ne100_ne300.json` va
  `results/runs/route_neighborhood_ne500.json`.
