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
