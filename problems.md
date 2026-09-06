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
- P-001 retrieval-window auditida M=8 → M=16/24 candidate oracle regret
  `0.1953/0.1710 → 0.1099/0.0744 → 0.0383/0.0377` ga tushdi, recall esa
  `9.9%/12.1% → 28.4%/28.2% → 64.2%/55.9%` bo‘ldi. Lekin ayni key-score
  selectorning selection regreti M=32 da `0.4878/0.5092` gacha oshdi. Demak
  retrieval haqiqiy bottleneck, ammo tor poolni shunchaki kengaytirish yechim
  emas; selection/objective mismatch ham mustaqil muammo.
- Frozen bankda real circuit output → GRU → immediate output proxy key-score'dan
  ancha yaxshi juftlik tanladi: M=8 proxy selection regreti seed17/18 uchun
  `0.079/0.036`, key-score esa `0.262/0.276`; M=32 proxy `0.118/0.060`,
  key-score `0.488/0.509`. Bu kuchli arxitektura signali, lekin proxy barcha
  circuit chiqishini diagnostic sifatida hisoblaydi va hali sparse inference
  yechimi emas.

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

**Qo‘shimcha cheklov:** M=16/24/32 counterfactual oracle natijalarini amaliy
quality improvement deb hisoblamang. Widened pool faqat retrieval headroomni
ko‘rsatadi; expert patchi candidate poolga kirgan circuitlardan final corrected
output uchun to‘g‘ri pairni tanlay olishini ham isbotlashi kerak.

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

## Expert uchun navbatdagi katta ishlar — to‘liq handoff

Quyidagi navbat expertga ishni qismlarga bo‘lib topshirish uchun yozildi. Har
bir topshiriq alohida branch yoki alohida opt-in experiment bo‘lsin. Bir nechta
aktiv problemni bir patchda aralashtirmang. Expert tayyorlagan kodni Codex
mustaqil test qiladi; benchmark gate bajarilmaguncha hech bir variant defaultga
aylantirilmaydi.

### Handoff A — P-001 candidate retrieval diagnostikasi va minimal tuzatish

Bu hozirgi eng katta va eng shoshilinch ish. Avval yangi router yozishdan ko‘ra,
candidate pool ichida kerakli circuitlar yo‘qolayotganini aniq ajrating.

#### O‘qilishi kerak bo‘lgan dalillar

- `results/runs/capacity_route_oracle_audit_s17_s18.json` va unga mos
  `results/` hisobotlari: learned route, candidate-pool oracle va full-bank
  oracle orasidagi headroom.
- `results/V0_179_FLAT_ROUTER_SCREEN.md`: barcha keylarni score qilishning o‘zi
  sifat bermaganini ko‘rsatuvchi negative control.
- `results/V0_180_PROBEROUTE2_FROZEN_BANK.md`: retrieval-only va
  selection-only o‘zgarishlari barqaror foyda bermagan.
- `results/COUPLED_PROBE_ROUTER_AUDIT.md`: router parametrlari kamaygan, CE
  yaxshilangan, lekin candidate recall tushgan va hard accuracy deyarli
  o‘zgarmagan.
- `results/P004_CASCADE_CREDIT_SEED17_18.md`: cascade-consistent credit CE va
  regretning ayrim komponentlarini yaxshilagan bo‘lsa-da, accuracy gate'larini
  bajarmagan.

#### Birinchi diagnostika talabi

Mavjud frozen checkpoint/bank va bir xil held-out inputlar ustida route
qarorlarini qayta ko‘rsating. Kamida quyidagi narsalar alohida o‘lchansin:

1. candidate poolga target/full-oracle pairning ikkala circuiti kirgan-kirmagani;
2. pair retrieval regreti — eng yaxshi full-bank pair candidate poolda yo‘q
   bo‘lgani uchun yo‘qotilgan qiymat;
3. candidate ichida eng yaxshi pairni tanlash regreti — yaxshi pair poolda bor,
   lekin selector noto‘g‘ri tanlagan holat;
4. har bir internal step va prefix/suffix cascade holati bo‘yicha ushbu ikki
   xatoning taqsimoti;
5. candidate size M o‘zgarganda (kamida M=8 va M=16, bank=32) recall va
   regretning qanday o‘zgarishi.

Bu audit “router yomon” degan umumiy xulosani representation, candidate size,
loss/target yoki cascade distribution muammolaridan biriga ajratishi kerak.
Counterfactual oracle hisobida final corrected output ta’rifi bir xil bo‘lsin;
native, fixed-learned, candidate oracle va full oracle raqamlarini aralashtirmang.

#### Patch faqat diagnostika sababini tasdiqlasa yozilsin

Minimal patchlardan faqat bittasini tanlang: candidate scoring signalini
to‘g‘rilash, candidate-pool size/schedule'ni o‘zgartirish yoki retrieval loss
targetini tuzatish. Pair selector, circuit body, correction weights va recurrent
state update'ni birinchi patchda birga o‘zgartirmang. Attention/Transformer
qo‘shmang; Neural Engine'ning sparse-circuit maqsadi saqlansin.

#### P-001 acceptance gate

- E=32, active=2, M=8, T=3, seed17/18 protocol eski benchmark bilan bir xil;
- candidate recall eski routerdan pasaymasin;
- p95 selection/retrieval regret kamida 10% yaxshilansin;
- held-out hard accuracy o‘rtacha kamida +2 pp;
- dead circuits `<=3/32`, inference latency `<=1.25x`;
- total, touched/active parameter va probe cost alohida ko‘rsatilsin;
- 5000-step full run bo‘lmasa, natijani final improvement deb yozmang.

Deliverable: kod, unit test, kichik smoke benchmark, seed17/18 full benchmark,
JSON/Markdown audit, exact reproduction command va `problems.md`dagi qaror.
Gate bajarilmasa `REJECTED` tarixiga natija, sabab va keyingi qaror yozilsin.

### Handoff B — P-005 CE–accuracy/regret objective diagnostikasi

P-001 retrieval mexanizmi aniqlanmaguncha katta loss rewrite qilmang. Keyin
P-005 uchun bir xil frozen route/circuit bank ustida mean CE, hard accuracy,
mean/p95 regret va exact pair oracle'ni birga hisoblang. CE-only improvementni
qabul qilmang. Agar loss patch sinalsa, eski loss bilan yangi lossni seed17/18
paired benchmarkda 2x2 qilib solishtiring; hard-selection va on-policy
trajectory alohida hisobot bo‘lsin. Target final corrected outputga mosligini
gradient/test bilan tekshiring. Minimal patch, opt-in, default o‘zgarmaydi.

### Handoff C — P-006 active-parameter va routing-cost instrumentation

Bu sifat eksperimenti emas. `parameter_report()` va benchmark hisobotlarini
model body, router projection, key-table read, candidate scoring, selector,
circuit body, recurrent step hamda correction bo‘yicha touched bound bilan
to‘ldiring. Training probe cost va inference routing costni ajrating. Formula
uchun unit test va eski benchmark JSONlariga backward-compatible maydonlar
qo‘shing. Instrumentation patchi quality modelini o‘zgartirmasin.

### Handoff D — local output-aware cost signalni sparse ko‘rinishga keltirish

P-001 diagnosticida immediate post-update output proxy oddiy key-score’dan
sezilarli yaxshi bo‘ldi. Bu natija faqat frozen counterfactual upper-bound emas:
u candidate pool ichida final pairni tanlash uchun foydali signal borligini
ko‘rsatadi. Lekin diagnostic proxy har bir bank circuitining real chiqishini
hisoblaydi; uni bevosita inference routerga qo‘yish Neural Engine’ning active
parameter maqsadini buzadi.

Expert task: avval proxy formulasi qaysi qismdan foyda olayotganini ajrating —
individual circuit output, pair interaction, immediate GRU state yoki output
head. Keyin faqat bitta sparse implementation taklif qiling: masalan,
oldindan o‘qitilgan compact output signature, candidate-only output probe yoki
shared low-rank summary. Barcha bankni inference vaqtida dense ishlatadigan
variantni yechim deb hisoblamang. Circuit body va default model birinchi
patchda o‘zgarmasin; retrieval va selector natijalari alohida ko‘rsatilsin.

Acceptance: seed17/18, E=32, active=2, T=3; candidate recall pasaymasin;
hard accuracy kamida `+2 pp`, mean/p95 regret kamida `10%` yaxshilansin;
latency `<=1.25x`; active parameters va probe cost to‘liq hisoblansin. Faqat
proxy oracle yaxshilangani adoption uchun yetarli emas. Kod, test, smoke/full
benchmark, JSON/Markdown audit va gate bajarilmasa negative report yozilsin.

### Expertga yuboriladigan ish tartibi

1. Avval faqat Handoff A diagnostikasini bajaring.
2. Diagnostic natijasi P-001 sababini ko‘rsatmasa, patch yozmang; failure report
   qoldiring va nimani o‘lchash yetishmaganini ayting.
3. Sabab aniq bo‘lsa, bitta minimal opt-in patch va paired benchmark yozing.
4. P-001 gate'lari bajarilmasa, uni `REJECTED` deb belgilang; P-005 yoki P-006ga
   faqat alohida branch/commitda o‘ting.
5. Har bir natijada “quality”, “retrieval”, “active cost” va “training cost”ni
   alohida jadvalda bering. Bitta CE raqami bilan yechimni tasdiqlamang.

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

### C-P004-CASCADE-001 — Cascade-consistent on-policy credit

**Status:** `REJECTED`
**Muammo:** P-004
**Natija:** seed17/18 held-out accuracy mean delta `-0.625 pp`, held-out CE delta `-0.00006`, mean regret reduction `-6.83%`, p95 regret reduction `-19.08%`, training overhead `1.291x`. Pre-registered gate bajarilmadi; P-004 `ACTIVE` qoladi.
**Batafsil:** `results/P004_CASCADE_CREDIT_SEED17_18.md` va `results/P004_CASCADE_CREDIT_SEED17_18.json`.

### C-P001-STRIDED-001 — Strided M=8 candidate schedule

**Status:** `REJECTED FOR ADOPTION`  
**Muammo:** P-001  
**Patch:** mavjud `routing_windows` kontrakti orqali M=8 candidate budgetini 4×2 bank windowga yoyadigan runtime-only opt-in schedule; key-score selector, circuit body, correction, recurrent update va default model o‘zgarmaydi. Yangi trainable parametr va training probe yo‘q.  
**Qaror sababi:** mavjud M=16/24 diagnostic retrievalni selectiondan ajratib berdi, lekin yangi strided M=8 patch uchun talab qilingan seed17/18 checkpointlar branch/release/Actions artifact sifatida mavjud emas. Shu sabab patchning hard accuracy, CE, p95 retrieval/selection regret, dead-circuit va latency gate'lari ushbu handoffda isbotlanmadi. Fail-closed acceptance bo‘yicha promotion rad qilindi; bu hypothesis ilmiy rad etildi degani emas. P-001 `ACTIVE` qoladi.  
**Batafsil:** `results/P001_STRIDED_RETRIEVAL_PATCH.md`; benchmark: `benchmark_p001_strided_retrieval.py`.
