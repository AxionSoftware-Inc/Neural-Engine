# Neural Engine — muammolar reyestri

Bu fayl loyihadagi eng muhim ochiq muammolar uchun yagona ishchi reyestrdir.
Expertga butun tarixni yuborish shart emas: kerakli problem ID va uning
`Expert task` qismini yuborish kifoya. Expert patch va benchmark tayyorlaydi,
keyin Codex uni mustaqil test qiladi.

## Ishlash qoidasi

- Har bir muammo alohida ID bilan yuritiladi.
- `ACTIVE` muammo yechilmaguncha shu yerda qoladi.
- Faqat test va benchmark bilan tasdiqlangan yechim `SOLVED` ga o‘tkaziladi.
- Ishlamagan taklif `REJECTED` sifatida arxivda saqlanadi; tarix o‘chirilmaydi.
- Bir paytda bitta asosiy muammoni o‘zgartirish kerak, aks holda sababni
  ajratib bo‘lmaydi.
- Har bir patch eski defaultni o‘zgartirmaydigan opt-in experiment sifatida
  boshlanadi.

## Aktiv muammolar

### P-001 — Candidate retrieval kerakli circuitlarni topmayapti

**Status:** `ACTIVE`  
**Prioritet:** P0  
**Ta’sir:** katta bankda sifatning oshmasligi va active parametrlar foydali
bo‘lmagan circuitlarga sarflanishi.

#### Dalil

- V0.178 oracle auditida learned candidate route bilan full-bank oracle
  orasida qo‘shimcha headroom bor: seed17 `0.195 CE`, seed18 `0.171 CE`.
- CoupledProbeRouter old ProbeRouteRouter bilan solishtirilganda candidate
  recall har ikki seedda `−8.33 pp` tushdi.
- FlatRouter barcha keylarni score qilganida ham sifat yaxshilanmadi; demak,
  faqat full-bank score qo‘shish yetarli emas.

#### Muammo ta’rifi

Retriever arzon bo‘lishi kerak, lekin yaxshi pair ichidagi circuitlarning
ikkalasini candidate poolga muntazam kiritishi kerak. Hozirgi retriever
signalida candidate inclusion va final pair quality o‘rtasida mismatch bor.

#### Qabul qilish mezonlari

- E=32, active=2, M=8, T=3, seed17/18 bir xil protocol;
- candidate recall old routerdan pasaymasin;
- p95 regret kamida 10% yaxshilansin;
- hard accuracy o‘rtacha kamida `+2 pp` bo‘lsin;
- dead circuits `<=3/32`, inference latency `<=1.25x`;
- router body/circuit API va default model buzilmasin.

#### Expert task

Faqat candidate retrieval muammosini yechadigan minimal router yoki retrieval
loss patchini yozing. Pair selector, circuit body va recurrent state update'ni
bir vaqtda o‘zgartirmang. Candidate recall, retrieval regret, selection regret,
hard accuracy va CE ni eski router bilan paired benchmarkda o‘lchang. Gate
bajarilmasa, taklifni `REJECTED` deb aniq belgilang.

---

### P-002 — Circuit bank specializationi yetarli emas

**Status:** `ACTIVE`  
**Prioritet:** P0  
**Ta’sir:** bank hajmi oshganda sifat oshmasligi yoki pasayishi; yaxshi router
ham foydali bo‘lmagan circuitlarni tanlashi.

#### Dalil

- FlatRouter candidate retrievalni olib tashladi, lekin 5000-step full screen
  `44.89%` bilan hierarchical reference `47.18%` dan yomon chiqdi.
- Bu natija muammo faqat router score'ida emasligini ko‘rsatadi: circuitlar
  sparse gradient orqali yetarli specializationga ega bo‘lmayapti.
- Capacity scaling tajribalarida bank kattalashishi muntazam quality scaling
  bermadi.

#### Muammo ta’rifi

Circuitlar alohida va qayta ishlatiladigan funksiyalarni o‘rganishi kerak.
Hozir route tanlovi bilan circuit learning signali bir-biriga bog‘langan:
kam tanlangan circuit o‘qimaydi, o‘qimagan circuit esa tanlanmaydi.

#### Yangi diagnostik dalil (2026-09-07)

100M/300M/500M staged 10k checkpointlarda candidate circuit chiqishlarining
mean pair cosine’i mos ravishda seed17/18 uchun `0.040/0.040`, `0.043/0.042`
va `0.042/0.041` bo‘ldi. Demak bank functional jihatdan to‘liq collapse
bo‘lmagan. Biroq ishlatilgan bank ulushi `44.5–45.6% → 16.1–16.3% →
12.4–12.6%` ga tushdi, used-circuit task entropy esa `~0.25 → ~0.14 →
~0.07` bo‘ldi. Bu capacity oshganda over-specialization va route
fragmentation kuchayayotganini ko‘rsatadi.

#### Qabul qilish mezonlari

- router o‘zgarmagan control bilan solishtirish;
- har bir circuitga tushgan gradient/forward/usage statistikasi;
- dead yoki under-trained circuitlar soni;
- 20M/32-bankda seed17/18 uchun kamida `+2 pp` accuracy yoki aniq
  specialization signali;
- training active compute va inference active parameters alohida hisoblansin.

#### Expert task

Routerga tegmasdan circuit bank specialization yoki sparse credit-assignment
uchun bitta minimal experiment yozing. Circuit body, correction va routing
gradientlarini qaysi tartibda va nima uchun o‘zgartirayotganingizni ko‘rsating.
Joint architecture rewrite qilmang. Natija bo‘lmasa, negative resultni va
qaysi gipoteza rad etilganini hujjatlashtiring.

**Diagnostika:** `results/P002_CIRCUIT_FUNCTIONAL_SPECIALIZATION_AUDIT.md`.
Keyingi sinov functional collapse’ni emas, circuitlararo qayta foydalanish va
route fragmentationni kamaytirishni tekshirishi kerak.

Tasklararo path-distribution variance regularizer (`routing_reuse_weight`) ham
sinab ko‘rildi. `weight=2.0` 100Mda all-screenni `86.48% → 86.22%` ga tushirib,
dead fractionni `7.15% → 32.27%` qildi. Yumshoq `weight=0.25` all-screenni
`+0.08 pp` oshirdi, lekin held-out active-8ni `−0.21 pp` pasaytirdi va route
replay sensitivity faqat `+0.42 pp` bo‘ldi. Oddiy reuse regularizeri
`REJECTED FOR ADOPTION`; P-002 ochiq qoladi.

**Audit:** `results/P008_ROUTING_REUSE_AUDIT.md`.

#### Keyingi opt-in gipoteza

Hierarchical routerdagi tasklararo path-distribution variance uchun yumshoq
`routing_reuse_weight` regularizer sinov qilinadi. U circuit ID yoki active
budgetni majburlamaydi; faqat katta bankda har taskning butunlay alohida
subtreega parchalanishini kamaytirishni ko‘zlaydi. Qabul qilish faqat held-out
hard accuracy, route coverage va active cost birga yaxshilansa mumkin.

---

### P-003 — Capacity oshganda quality scaling kafolatlanmayapti

**Status:** `ACTIVE`  
**Prioritet:** P0  
**Bog‘liq:** P-001, P-002, P-004

#### Dalil

20M/50M/100M/300M yo‘nalishlarida parametr hajmi oshishi sifatni muntazam
oshirmadi; ayrim testlarda sifat o‘zgarmadi yoki pasaydi. Shuning uchun hozir
katta modelga o‘tish ilmiy jihatdan asoslanmagan.

#### Muammo ta’rifi

Model capacity'si ko‘payganda yangi parametrlar foydali, kirish-dependent
funksiyalarga aylanishi va inference'da kerakli subsetga aylanishi kerak.
Hozir capacity ko‘payishi bilan routing, specialization yoki optimization
muammolari kuchaymoqda.

#### Qabul qilish mezonlari

- bir xil task, data, active budget va evaluation protocol;
- 20M → 50M → 100M kamida monotonic trend yoki aniq scaling law;
- active parameters sekin o‘sishi, total parameters esa o‘sishi;
- quality pasaysa, sabab P-001/P-002/P-004 bilan ajratib berilsin.

#### Expert task

Yangi katta model yozmang. Avval kichik banklarda capacity scalingni buzayotgan
aniq mexanizmni topadigan diagnostic experiment yozing. Natija faqat “model
kattaroq bo‘ldi” emas, qaysi circuitlar yangi capacity'dan foyda olayotganini
ko‘rsatsin.

---

### P-004 — Sparse training credit assignment va cascade shift

**Status:** `ACTIVE`  
**Prioritet:** P1  
**Bog‘liq:** P-001, P-002

#### Dalil

- Router o‘zgarganda keyingi recurrent step input taqsimoti ham o‘zgaradi.
- Frozen-bank ProbeRoute-2 full rejimida CE o‘rtacha yaxshilangan, ammo hard
  accuracy seedlar bo‘yicha barqaror yaxshilanmagan.
- Selection-only va retrieval-only control'lar alohida ishonchli foyda bermadi.

#### Muammo ta’rifi

Bir step uchun tanlangan circuitning foydasi keyingi step state'iga bog‘liq.
Training faqat mahalliy route signaliga yopishsa, butun cascade uchun foydali
bo‘lmagan circuit tanlanishi mumkin. Shu bilan birga kam tanlangan circuitga
gradient yetib bormaydi.

#### Qabul qilish mezonlari

- prefix, changed step va suffix alohida hisobga olinsin;
- final corrected output asosidagi target ishlatilsin;
- on-policy va held-out hard accuracy birga yaxshilansin;
- route va circuit gradient/usage statistikasi chiqarilsin;
- probe cost training va inference cost alohida ko‘rsatilishi kerak.

#### Expert task

Faqat credit-assignment yoki cascade data aggregation uchun minimal training
patch yozing. Router arxitekturasini yangidan ixtiro qilmang. Frozen-bank
control va on-policy controlni alohida ko‘rsating; negative natijani ham
saqlang.

---

### P-005 — CE yaxshilanishi hard accuracy'ga aylanmayapti

**Status:** `ACTIVE`  
**Prioritet:** P1  
**Bog‘liq:** P-001, P-004

#### Dalil

CoupledProbeRouter old routerga qaraganda o‘rtacha CE'ni `0.05357` ga
yaxshiladi, lekin hard accuracy faqat `+0.169 pp` bo‘ldi. Ba’zi rejimlarda CE
kamayib, accuracy pasaydi.

#### Muammo ta’rifi

Router tanlovi o‘rtacha lossni pasaytirayotgan bo‘lishi mumkin, lekin to‘g‘ri
class margin yoki eng yaxshi pairni tanlashni yaxshilamayapti. Demak, loss
target, ranking va evaluation metric o‘rtasida mismatch bo‘lishi mumkin.

#### Qabul qilish mezonlari

- mean CE, hard accuracy, mean/p95 regret va candidate recall bir jadvalda;
- exact pair oracle alohida va on-policy trajectory oracle alohida;
- kamida seed17/18, imkon bo‘lsa seed19;
- faqat CE yaxshilangani adoption uchun yetarli hisoblanmasin.

#### Expert task

Yangi router yozishdan oldin objective/evaluation mismatchni tekshiradigan
minimal diagnostic yoki loss patch yozing. Hard accuracy va regretga zarar
beradigan CE-only improvementni avtomatik qabul qilmang.

---

### P-006 — Active parameter va routing cost hisoboti to‘liq emas

**Status:** `ACTIVE`  
**Prioritet:** P2  
**Bog‘liq:** barcha routing tajribalari

#### Dalil

`NeuralEngineV0.parameter_report()` yangi probe router projection va key
table'larining barcha touched qismini active estimate'ga to‘liq kiritmaydi.
CoupledProbe benchmark konservativ alohida hisob ishlatishga majbur bo‘ldi.

#### Muammo ta’rifi

“Faqat kerakli parametrlar ishladi” degan da’vo router score, shared controller,
candidate key read, circuit body va recurrent step bo‘yicha bir xil metod bilan
hisoblanishi kerak. Aks holda active parameter va latency natijalari noto‘g‘ri
taqqoslanadi.

#### Qabul qilish mezonlari

- har bir router variant uchun touched parameter bound;
- total parameters, active parameters/decision va active parameters/example;
- inference routing cost va training probe cost alohida;
- kamida unit test bilan formula tekshirilishi.

#### Expert task

Model body yoki router sifatini o‘zgartirmasdan parameter/cost instrumentation
patchini yozing. Eski testlarni saqlang va oldingi benchmark natijalari bilan
backward-compatible hisobot chiqaring.

## Yopilgan yoki rad qilingan yo‘llar

Bu bo‘lim aktiv muammolarni to‘ldiradi; muvaffaqiyatsiz tajribalar o‘chirilmaydi.

### C-001 — Flat full-bank router

**Status:** `REJECTED`  
**Natija:** 5000-step full screen mean `44.89%`, hierarchical reference `47.18%`
dan yomon; dead circuits `34.4%/37.5%`. Barcha keylarni score qilishning o‘zi
muammoni hal qilmadi. Batafsil: `results/V0_179_FLAT_ROUTER_SCREEN.md`.

### C-002 — ProbeRoute-2 frozen-bank router

**Status:** `REJECTED FOR ADOPTION`  
**Natija:** full rejim CE o‘rtacha yaxshilangan, lekin accuracy o‘rtacha
`−0.21 pp`; selection-only va retrieval-only barqaror foyda bermadi. Batafsil:
`results/V0_180_PROBEROUTE2_FROZEN_BANK.md`.

### C-003 — CoupledProbeRouter compact variant

**Status:** `REJECTED FOR DEFAULT`  
**Natija:** router `42,240 → 5,632` parametrga tushdi va CE o‘rtacha
`−0.05357` yaxshilandi, lekin accuracy faqat `+0.169 pp`, candidate recall esa
har ikki seedda `−8.33 pp` bo‘ldi. Batafsil:
`results/COUPLED_PROBE_ROUTER_AUDIT.md`.

## Expertga yuborish uchun qisqa format

```text
Neural Engine repo'dagi problems.md faylini o‘qing.
Faqat P-XXX muammosi bilan ishlang. Boshqa aktiv muammolarni bir vaqtda
o‘zgartirmang. Mavjud defaultni almashtirmaydigan minimal patch yozing.
Oldingi experimentlar va acceptance gate'larini saqlang. Kod, test, benchmark,
reproduction command va natijani yozing. Gate bajarilmasa, yechimni rad eting
va nima sabab ishlamaganini ko‘rsating.
```

### C-P002-CRCA-001 — Causal responsibility sparse credit

**Status:** `REJECTED`
**Muammo:** P-002
**Natija:** seed17/18 held-out mean accuracy delta `+0.052 pp`, held-out counterfactual NMI delta `-0.09967`, specialization delta `-0.12178`, positive final-CE advantage delta `+0.89793`. Pre-registered P-002 gate bajarilmadi; CRCA default uchun qabul qilinmadi.
**Batafsil:** `results/P002_CRCA_SEED17_18.md` va `results/P002_CRCA_SEED17_18.json`.
Codex mustaqil qayta ishlatgan to‘liq CUDA benchmark ham ayni qaror va raqamlarni tasdiqladi; batafsil: `results/P002_CRCA_RECHECK_20260907.md`.

### C-P003-PROGRESSIVE-CAPACITY-001 — Progressive bank exposure

**Status:** `REJECTED FOR DEFAULT`  
**Muammo:** P-003
**Natija:** NE-50 ikki seed mean accuracy `71.54% → 71.11%` (`−0.43 pp`), NE-100 `71.69% → 72.10%` (`+0.40 pp`). NE-100 dead fraction biroz kamaydi, active parametrlar `~1.98M`da qoldi, lekin +2 pp gate bajarilmadi va NE-50 seed18 regressiya qildi. P-003 `ACTIVE` qoladi.
**Batafsil:** `results/P003_PROGRESSIVE_CAPACITY_SEED17_18.md`.

### C-P003-MATCHED-SCALE-001 — Matched 20M/50M/100M capacity screen

**Status:** `REJECTED FOR SCALING`  
**Muammo:** P-003
**Natija:** Bir xil 5,000-step, balanced, coverage-aware trainingda ikki seed mean accuracy NE-20 `72.06%`, NE-50 `71.54%`, NE-100 `71.69%` bo‘ldi. NE-50/100 NE-20dan mos ravishda `−0.52/−0.36 pp`; dead fraction `0.18% → 6.28%/6.61%`. P-003 `ACTIVE` qoladi: warmup, longer learning curve va clean held-out matched training hali kerak.
**Batafsil:** `results/P003_MATCHED_CAPACITY_SEED17_18.md`.

### C-P003-10K-SCALE-001 — Longer-budget capacity continuation

**Status:** `REJECTED FOR SCALING`
**Muammo:** P-003
**Natija:** 10,000-step, uch-seed matched screenda NE-20 direct mean
`77.99%`, NE-100 progressive mean `77.86%` bo‘ldi (`−0.13 pp`). Seedlar
bo‘yicha farq `+0.29 / −0.99 / +0.31 pp`; seed18 regressiyasi capacity foydasi
barqaror emasligini ko‘rsatadi. NE-100 progressive NE-100 direct 2-seed
controldan `+0.49 pp` yuqori bo‘lsa-da, bu schedule foydasi bo‘lib, NE-20dan
ustun capacity dalili emas. 20M modelning 5k→10k o‘sishi `+6.56 pp` bo‘lib,
oldingi screen undertraining ta’sirida bo‘lganini ko‘rsatdi. P-003 `ACTIVE`
qoladi: routing/circuit utilization va murakkab task regressiyalari tekshirilishi
kerak.
**Batafsil:** `results/P003_MATCHED_10K_SEED17_19.md`.

### C-P003-STAGED-GROWTH-001 — Inherited circuit bank with staged exposure

**Status:** `PROMISING — SCALE SATURATION OBSERVED`
**Muammo:** P-003
**Natija:** NE-20 5k checkpointdan parent circuit/router weightlari ko‘chirilib,
avval kichik reachable bankda, keyin full bankda training qilindi. 100M staged
growth seed17/18/19 full-bank mean accuracy `82.42%` va clean held-out mean
`82.14%` berdi. Keyingi 300M/500M scale auditida full stage 10k exposure bilan
300M mean `84.99%`, 500M mean `84.94%` bo‘ldi; 300M/500M active-8 held-out
mean mos ravishda `85.05%/85.05%` bo‘ldi. NE-20 direct 10k mean `77.99%`dan
katta ustunlik saqlanadi, lekin 300Mdan 500Mga qo‘shimcha foyda yo‘q.
Staged growth hanuz eng kuchli yo‘l, ammo parametr sonini oshirish bilan emas,
full-bank exposure va route sifati bilan foyda bergan; shuning uchun `SOLVED`
yoki default emas.
**Batafsil:** `results/P003_STAGED_BANK_GROWTH_SEED17_18.md` va
`results/P003_STAGED_SCALE_300M_500M_SEED17_18.md`.

### C-P003-DIRECT-GROWTH-001 — Direct inherited growth control

**Status:** `REJECTED`
**Muammo:** P-003
**Natija:** NE-20 5k checkpointdan full 7552 bankka to‘g‘ridan-to‘g‘ri ko‘chirish
va yana 5k training seed17/18da `77.76% / 78.54%`, mean `78.15%` berdi.
Bu NE-20 direct 10k mean `78.42%`dan yuqori emas va staged exposure bo‘lmasa
katta sakrash yo‘q. Demak foyda oddiy weight-copy emas; avval kichik reachable
bankda moslashish bosqichi kerak bo‘lishi mumkin.
**Batafsil:** `results/P003_STAGED_BANK_GROWTH_SEED17_18.md`.

### C-P003-ROUTE-FRAGMENTATION-001 — Large-bank route fragmentation

**Status:** `ACTIVE`
**Muammo:** P-003
**Dalil:** 10k route auditida noldan NE-100 progressive variantlarida task
ichidagi route Jaccard `0.0049–0.0112`, eval dead fraction `31.70–34.18%` va
between-task Jaccard `0.0480–0.0522` bo‘ldi. Staged growthda task route’lari
ko‘proq qayta ishlatildi/dead fraction `15.06–16.72%`gacha tushdi va quality
`+4.40 pp` o‘sdi. Bu routing fragmentationni kuchli nomzod qiladi, lekin
staged growthdagi qo‘shimcha training va meros qilingan circuitlar ta’siri
hali alohida ajratilmagan.
**Keyingi tajriba:** route fragmentation/coverage’ni kamaytiruvchi minimal patchni
100M va 300M control bilan tekshirish; 500M faqat scale-control sifatida qoladi.

### C-P003-SCALE-SATURATION-001 — Capacity growth does not improve held-out quality

**Status:** `ACTIVE`
**Muammo:** P-003
**Dalil:** Bir xil staged full-bank exposure va held-out active-budget
evaluatorida 100M/300M/500M 10k active-8 means `84.90% / 85.05% / 85.05%`.
All-screen means `84.99% / 84.99% / 84.94%`. 500M total parametrni 100Mga
nisbatan 5x oshirdi, active decision esa `~1.98M`da qoldi; 500M eval route
auditida dead circuit ulushi `60.26–60.97%` bo‘ldi.

**Muammo ta’rifi:** Bank kattalashmoqda, lekin yangi circuitlar route orqali
yetarli darajada foydali va qayta ishlatiladigan computationga aylanmayapti.
Shuning uchun qo‘shimcha parametrlarning katta qismi sifatga aylanmasdan
unused/fragmented bankda qolmoqda. Bu fundamental arxitektura imkonsizligini
isbotlamaydi, ammo yana 700M/1B scale’ga o‘tishdan oldin hal qilinishi kerak.

**Qabul qilish mezonlari:**

- 100M/300M/500M bilan bir xil data, active budget, seed va held-out evaluator;
- kamida ikki seedda active-8 mean uchun `>=+1 pp` yoki murakkab tasklarda
  barqaror `>=+3 pp` improvement;
- route dead fraction va task route overlap hisobotlari;
- total/active params, latency va VRAM alohida qayd qilinsin.

**Keyingi tajriba:** yangi bankni kattalashtirmasdan route fragmentation/coverage
muammosini minimal patch bilan sinash. 500M konfiguratsiya scale-control sifatida
saqlanadi, defaultga ko‘chirilmaydi.

**Batafsil:** `results/P003_STAGED_SCALE_300M_500M_SEED17_18.md`.

### P-007 — Selected circuit route outputni yetarli boshqarmayapti

**Status:** `ACTIVE`
**Prioritet:** P0
**Bog‘liq:** P-001, P-002, P-003, P-004

#### Dalil

- 100M/300M/500M 10k checkpointlarda circuit delta normasi shared encoded
  signalining faqat `~3.9–4.9%`iga teng bo‘ldi.
- 100% route replayda seed17 global accuracy drop 100M/300M/500M uchun
  `−0.26/−0.52/−0.16 pp`; within-task drop `−0.31/−0.05/−0.05 pp` bo‘ldi.
- 500M seed18da global drop `+0.05 pp`, within-task drop `0.00 pp` bo‘ldi.
- 500M seed17da route almashtirish circuit delta’ni encoded normaning
  `5.2–5.9%`i miqdorida o‘zgartirdi, lekin final hard accuracy sezilarli
  o‘zgarmadi.

#### Muammo ta’rifi

Router circuit tanlayapti, ammo tanlangan circuitning correction yo‘li shared
input reinjection/recurrent state ichida juda kichik ulushga ega bo‘lishi yoki
circuitlar bir-biriga o‘xshash funksiyani berishi mumkin. Bunday holatda
candidate retrievalni yaxshilashning o‘zi yetarli emas: route qarori causal
bo‘lmasa, qo‘shimcha capacity sifatga aylanmaydi.

#### Qabul qilish mezonlari

- kamida ikki seedda held-out hard accuracy baseline’dan pasaymasin;
- 100% global va within-task route replayda natural routega nisbatan kamida
  `+1 pp` accuracy drop yoki equivalent logit/state sensitivity ko‘rinsin;
- circuit delta/encoded normasi o‘lchansin va faqat forced gain bilan sun’iy
  ko‘tarilmasin;
- active params, latency va training stability alohida hisobot qilinsin;
- default model o‘zgarmasin, patch opt-in bo‘lsin.

#### Keyingi tajriba

Router arxitekturasini birdan almashtirmasdan, circuit correctionning state
update’dagi ulushini nazorat qiluvchi minimal variantni (learned bounded
correction gate yoki route-conditioned normalization) 100M ikki-seed controlda
sinash. Agar route replay sensitivity oshib, held-out quality yaxshilanmasa,
muammo circuit identifikatsiyasi/specialization tomoniga ko‘chiriladi.

**Batafsil:** `results/P003_STAGED_SCALE_300M_500M_SEED17_18.md`.

Scale=0.5 statik correction patchi sinab ko‘rildi: seed17 all-screen
`85.57% → 85.18%`, held-out active-8 `86.20% → 86.20%`. Test-time sweepdagi
kichik seed17 foydasi trainingga ko‘chmadi; bu patch `REJECTED FOR ADOPTION`.
Keyingi variant statik scale emas, learned bounded gate yoki route-conditioned
normalization bo‘ladi.

**Scale audit:** `results/P007_CORRECTION_SCALE_05_AUDIT.md`.

Learned bounded gate (`correction_gate_mode=route_bounded`) ham seed17da
tekshirildi: full 10k all-screen `85.57% → 85.29%`, held-out active-8
`86.20% → 86.25%`, route replay drop esa global/within-task `+0.05/+0.10 pp`
bo‘ldi. Quality yoki route sensitivity gate bajarilmadi; patch
`REJECTED FOR ADOPTION`.

**Gate audit:** `results/P007_CORRECTION_GATE_AUDIT.md`.
