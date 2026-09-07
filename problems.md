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

2026-09-07 counterfactual screen’da mavjud `router.keys` bilan full reachable
bank top-8 route qilindi. 64 misol/task, ikki seedda 100M/300M/500M accuracy
mean delta `−0.05/−0.26/−0.05 pp` bo‘ldi va CE barcha scale’da yomonlashdi.
Shuning uchun full-bank scoring hierarchical candidate retrievalning oddiy
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

**V0.185 representation audit (2026-09-07):** Qwen teacher gate/value
activation covariance asosida yangi 6.25%--12.5% compact SwiGLU basis qurildi.
Qiyin held-out protokolda teacher-derived variant `+0.1123`/`+0.1333` CE bilan
gate'dan o'tmadi; same-width random controls `+0.1067`/`+0.0972` bo'ldi.
25% va 50% random compact controls ham `+0.0790` va `+0.0832` bo'lib,
monotonic quality scaling bermadi. `V0.185`dagi activation-covariance
Galerkin basis `REJECTED`; bu Qwen'dan teacher-derived basislar umuman
imkonsiz degani emas, faqat shu projection formulasi yopildi.

**Audit:** `results/V0_185_QWEN_TEACHER_DERIVED_BASIS_SWIGLU.md`.

**V0.186 conditional-capacity oracle (2026-09-07):** 192/768/1536-wide
compact children bilan token difficulty bo'yicha teacher-informed width route
qilindi. Eng yaxshi oracle `34.03%` average active widthda ham `+0.0636` CE,
`25.49%`da `+0.0679` CE berdi; `<=50%` active budget gate'i bajarilmadi.
Demak dynamic width g'oyasi route diversity ko'rsatdi, lekin shu compact
functions representation gapini yopmadi; deployable difficulty predictor
ustida davom etish hozircha asoslanmagan.

**Audit:** `results/V0_186_QWEN_ADAPTIVE_WIDTH_ORACLE.md`.

**V0.187 signed group-sketch router (2026-09-07):** Qwen E=8/K=4 raw-neuron
bank uchun har bir group output'ining 8-D signed cosine sketch'i mavjud
pairwise cost headga berildi. Seed2026/27 learned CE `+0.05899/+0.05469`,
timing `1.183x/1.182x`, exact subset match esa layer25/26da taxminan
`53%/72%` bo'ldi. Old hidden-input pairwise controldan yaxshilanmadi va probe
cost oshdi. `REJECTED`; sketch dimensionni oshirish yoki 8-layer scale run
qilinmaydi.

**Audit:** `results/V0_187_QWEN_SIGNED_GROUP_SKETCH_ROUTER.md`.

**V0.188 dispatch audit (2026-09-07):** 8-layer Qwen E=8/K=6 runtime smoke’da
grouped/token-loop/packed backendlar mos ravishda `2.135x/2.125x/2.134x`
sekin chiqdi. Packedning birinchi token-wise weight gather varianti 9 GB
qo'shimcha VRAM so'rab OOM bo'ldi; per-group batched `F.linear`ga tuzatilib,
token-loop bilan `1e-6` parity berdi, lekin tezlashmadi. `torch.compile` esa
Windows buildda Triton yo'q bo'lgani uchun timinggacha yetmadi. P-006 ochiq;
haqiqiy speedup uchun fused CUDA/Triton backend kerak.

**Audit:** `results/V0_188_QWEN_DISPATCH_BACKEND_AUDIT.md`.

**V0.189 fused-dispatch audit (2026-09-07):** CUDA toolkit va Visual Studio
Build Tools mavjud muhitda opt-in `packed-fused` va custom `fused` backendlar
qo‘shildi. Gate+value projectionni bitta GEMMga birlashtirish 1-layerda
`269.78 → 269.71 ms` (ikkalasi `1.139x`) bo‘ldi; 8-layer qisqa smoke
`509.91 ms`, ya’ni V0.188 packed `505.88 ms`dan yaxshi emas. Custom kernel
real shape (`N=1024,H=1024,E=8,K=6,group=384`)da `832.6 ms`/dispatch chiqdi:
parity bor, lekin cuBLAS GEMMdan juda sekin. `REJECTED FOR ADOPTION`; default
o‘zgarmadi. P-006 uchun keyingi haqiqiy yo‘l tiled/grouped GEMM (CUTLASS,
cuBLAS grouped yoki Triton-capable environment), oddiy per-pair kernel emas.

**Audit:** `results/V0_189_QWEN_FUSED_DISPATCH_AUDIT.md`.

**V0.190 FP16 dispatch audit (2026-09-07):** Selected group GEMMlarni RTX
3060 Tensor Core uchun FP16ga o'tkazish 1-layer timingni `269.78 → 268.25 ms`
(`1.139x → 1.134x`) qildi. 8-layerda `494.61 ms / 238.62 ms = 2.073x`;
packed float32 control `2.134x` edi, demak faqat kichik (~3%) runtime foyda
bor, dense'dan hali ham sekin. Qisqa quality control alpha=0 CE `+0.0864`
bo‘lib gate’dan o‘tmadi. `REJECTED FOR ADOPTION`; default float32 grouped
qoladi. Keyingi speed yo‘li haqiqiy tiled/grouped GEMM.

**Audit:** `results/V0_190_QWEN_FP16_DISPATCH_AUDIT.md`.

**V0.191 dispatch-stage profile (2026-09-07):** To‘g‘ri `eval()` hard-path
profileda isolated Qwen layer26 MLP dense parent `3.935 ms`, grouped bank
`7.564 ms` (`1.92x`), packed bank `10.228 ms` (`2.60x`) chiqdi. Grouped uchta
batched projectionning o‘zi `2.457 ms`; qolgan katta ulush sort/argsort,
index gather/copy/add va reshape metadata’ga ketadi. Demak Python loopni yoki
precisionni almashtirish kifoya emas: ragged token grouping + tiled grouped
GEMM + accumulation bir kernelda birlashishi kerak. P-006 ochiq, V0.191
quality/capacity dalili emas.

**Audit:** `results/V0_191_QWEN_DISPATCH_STAGE_PROFILE.md`.

**V0.192 grouped-fused audit (2026-09-07):** Gate va value projectionlarini
bitta batched GEMMga birlashtirish isolated layer26 MLPni `7.564 → 6.549 ms`
qildi, ammo 8-layer end-to-end timing `506.93 ms / 238.88 ms = 2.122x` bo‘ldi;
V0.188 grouped control `506.60 ms / 237.27 ms = 2.135x` bilan amalda teng.
CPU parity `1e-6`dan o‘tdi, lekin launch sonini kamaytirish deployment speedup
bermadi. `REJECTED FOR ADOPTION`; P-006 uchun uch bosqichni (ragged grouping,
tiled GEMM, scatter) bitta kernelda boshqaradigan backend kerak.

**Audit:** `results/V0_192_QWEN_GROUPED_FUSED_AUDIT.md`.

**V0.193 correction-dispatch result (2026-09-07):** The hard rank-64
cross-group correction no longer gathers `[tokens,K,rank,hidden]` weights.
Expert-packed selected-token accumulation is mathematically equivalent at
`1e-6` and passes the full V0.174 quality control on seeds 2026/2027
(`+0.01141`/`+0.01837` CE). The comparable sparse timing falls from
`506.60 ms` to `268.42 ms` (`2.135x` to `1.128x` dense); trained runs measure
`0.991x` and `1.004x`. The rank-64 correction memory/dispatch bottleneck is
therefore solved for this tested path. P-006 remains active for unified
parameter/cost accounting, and the general K=4 router/capacity problems are
not solved by this systems patch.

**Audit:** `results/V0_193_QWEN_CORRECTION_DISPATCH_AUDIT.md`.

K=5 ham qayta tasdiqlandi: ikki seedli CE `+0.03881/+0.04036` bilan gate’dan
o‘tdi, timing esa avvalgi `1.91x`dan `1.134x/1.129x`ga tushdi. Shu sabab K=5
(`62.5%` active) hozirgi pastroq-budget operating point, K=6 esa yuqori
marginli reference sifatida saqlanadi.

V0.193 adaptive gather guard decode-like 1×32 timingni K=5’da `1.870x →
1.406x`, K=6’da `1.942x → 1.612x` qildi. Kichik batchda hali dense’ga teng emas;
compiled decode kernel/P-006 cost hisobi keyingi systems ishidir.
K=5 low-batch backend A/B’da grouped `1.406x`, grouped-fused `1.511x`, packed
`2.090x` bo‘ldi; oddiy backend almashtirish rad qilindi.
Haqiqiy one-token decode smoke’da (`1×1`, 100 iteration) K=5/K=6 `1.371x /
1.403x` bo‘ldi. Bu latency-only natija; seq=1 CE hisoblanmaydi. Compiled
decode dispatch hali ochiq.

**V0.194 K=4 pairwise aggregate (2026-09-07):** 36-component
`pairwise-cost-router`ni 3 round on-policy cascade refit bilan 8 qatlamda
sinash seed2026/2027’da `+0.06822/+0.07745` CE berdi; direct-hard subset
baseline `+0.06462/+0.06165` edi. O‘rtacha local regret `0.07392/0.07224`
bo‘lib qoldi. Demak richer pairwise head + aggregation route gapni tuzatmadi;
`REJECTED`. **Audit:** `results/V0_194_QWEN_K4_ON_POLICY_PAIRWISE_AUDIT.md`.

Deep-level reuse (`routing_reuse_start_level=2`, weight `2.0`) ham alohida
tekshirildi: all-screen controlga nisbatan faqat `+0.04 pp`, held-out active-8
esa `−0.47 pp`, route replay sensitivity esa deyarli oshmadi. Oddiy task-reuse
loss oilasi adoption uchun rad qilindi.

**Deep audit:** `results/P008_ROUTING_REUSE_AUDIT.md`.

Input reinjection schedule ham inference-only tekshirildi. Yumshoq
`[1,0.75,0.5]` schedule 64 misol/task, seed17/18 bo‘yicha o‘rtacha faqat
`+0.10 pp` accuracy berdi, CE esa `+0.0114` yomonlashdi; kuchliroq schedule’lar
barqaror regressiya qildi. Shuning uchun bu yo‘l trainingga o‘tkazilmadi va
`REJECTED FOR ADOPTION` deb belgilandi.

**Audit:** `results/P007_INPUT_REINJECTION_SCHEDULE_AUDIT.md`.

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

### C-P001-OUTPUT-SIGNATURE-FOLLOWUP-001 — Compact output-signature capacity and bank initialization

**Status:** `REJECTED FOR ADOPTION`  
**Muammo:** P-001 / Handoff D  
**Natija:** Random rank4/dim16 ikki-seed mean hard accuracy `−0.547 pp`,
random rank8/dim32 `−0.026 pp`; bank-init rank8/dim16 trainingga ochiq variant
`+0.469 pp` berdi, lekin mean/p95 selection-regret reduction faqat
`6.77%/9.43%` bo‘ldi. Signature’ni muzlatish `−0.651 pp` berdi. Individual-
additive target p95 regretni `12.16%`ga ko‘tardi, ammo final accuracy
`−0.234 pp` va max latency `1.312x` bo‘ldi. Bank-init trainable variant eng
yaxshi signal sifatida qayd qilindi, ammo `+2 pp` accuracy va `>=10%` regret
gate’lari bajarilmadi. Key-score prior `0.25` bilan qo‘shilganda ham accuracy
`−0.313 pp`, regret reduction `3.14%/7.77%` bo‘ldi. P-001 `ACTIVE` qoladi.
**Batafsil:** `results/P001_SPARSE_OUTPUT_SIGNATURE_FOLLOWUP_AUDIT.md`.

### C-P004-QWEN-TRANSFER-RECHECK-001 — Exact Qwen FFN compilation

**Status:** `VALIDATED CONTROL`
**Muammo:** P-004 bilan bog‘liq Qwen transfer lane
**Natija:** Lokal Qwen3-0.6Bning 28 ta MLP qatlami Neural Engine SwiGLU
circuitlariga training va calibration’siz ko‘chirildi. Max layer MLP error va
full-model max logit error `0.0`; converted parameter count `264,241,152`.
Bu tayyor FFN funksiyasidan boshlash mumkinligini tasdiqlaydi, ammo attention
bloklari hali qoladi va dense transfer active compute’ni kamaytirmaydi. Sparse
micro-group decomposition hamda held-out text quality muammosi ochiq.
**Batafsil:** `results/P004_QWEN_EXACT_TRANSFER_RECHECK_20260907.md`.

### C-P004-QWEN-CONTRIB-CLUSTER-001 — Output-space contribution clustering

**Status:** `REJECTED`
**Muammo:** Qwen sparse decomposition / P-004 transfer lane
**Natija:** Qwen contribution signature (`activation × down_proj`) bilan
balanced 8-group partition qilindi. 2-layer, top-4/8, rank-64 cross-group
smoke learned routerda `alpha=0 CE +0.1399`, exact best-subset oracle’da esa
`+0.1779` berdi; ikkalasi ham `+0.05` gate’dan o‘tdi emas. Oracle ham yomon
bo‘lgani uchun bu variantda asosiy bottleneck router emas, group decomposition
va missing signed contributions. 4-layer run qilinmadi.
**Batafsil:** `results/V0_181_QWEN_CONTRIBUTION_CLUSTER.md`.

### C-P004-QWEN-SIGNED-SUBSET-001 — Signed subset reconstruction

**Status:** `REJECTED FOR ADOPTION`
**Muammo:** Qwen sparse decomposition / P-004 transfer lane
**Natija:** Har bir `E=8, K=4` disjoint group subset uchun teacher-fitted
signed coefficient qo‘llandi. Exact cost oracle ikki seedda bir xil `+0.0374`
CE, 4-layer oracle `+0.0435` berdi; bu mavjud exact-subset control’dan katta
ustunlik emas. Learned subset router `+0.0720`, 300-step soft target `+0.0660`,
group-energy `+0.1088`, pairwise cost `+0.0659`, va 512-hidden router `+0.0642`
bo‘lib, hammasi `+0.05` gate’dan o‘tdi emas. Demak fixed `E/K` scale yagona
muammo emas; static signed reconstruction oracle’da barqaror bo‘lsa ham,
route’ni o‘rganish va decomposition bottleneckini hal qilmadi. True overlapping
codebook hali alohida gipoteza; 700M/1B scale bu natija asosida boshlanmaydi.
**Batafsil:** `results/V0_182_QWEN_SIGNED_SUBSET_RECONSTRUCTION.md`.

### C-P004-QWEN-CORE-OVERLAP-001 — Deterministic core-overlap codebook

**Status:** `REJECTED`
**Muammo:** Qwen sparse decomposition / P-004 transfer lane
**Natija:** High-energy core neuronlarini har bir group’da takrorlab, qolgan
tail’ni deterministic interleave qiluvchi `E=8,K=4` codebook sinab ko‘rildi.
Teacher-fitted signed subset oracle 25% core’da `+0.1429` CE, 12.5% core’da
`+0.1500` berdi; disjoint signed oracle `+0.0374` edi. Oraclening o‘zi
`+0.05` gate’dan o‘tmagani uchun learned router va 4-layer run qilinmadi.
Takrorlangan muhim neuronlar tashlab yuborilgan tail hissasini tiklamadi.
**Batafsil:** `results/V0_183_QWEN_CORE_OVERLAP_CODEBOOK.md`.

### C-P004-QWEN-CONTRIB-DIVERSE-001 — Contribution-diverse disjoint groups

**Status:** `REJECTED FOR ADOPTION`
**Muammo:** Qwen sparse decomposition / P-004 transfer lane
**Natija:** Output-space contribution clusterlarining teng bo‘laklari barcha
disjoint group’lar orasida tarqatildi. Exact oracle `+0.0397` CE berdi, bu
disjoint signed control `+0.0374`dan faqat `+0.0023`; learned router esa
`+0.0697` bilan gate’dan o‘tmadi. Shuning uchun bu layout katta siljish emas,
4-layer yoki 700M/1B davom ettirishga asos yo‘q.
**Batafsil:** `results/V0_184_QWEN_CONTRIBUTION_DIVERSE.md`.

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

### C-P003-TYPED-700M-REPLICATION-001 — Second-seed typed-register 700M validation

**Status:** `VALIDATED CONTROL — NO ADDITIONAL CAPACITY GAIN`
**Muammo:** P-003
**Natija:** Factorized typed-register parent-growth 700M modeli seed18da
mustaqil qayta tiklandi. Qat’iy `64^3 x 9` full-grid accuracy seed17/18da
`99.6627% / 99.5884%`, ikki seed mean `99.6255%` bo‘ldi. 500M parent-growth
mean `99.6622%` edi; 700M farqi `−0.0367 pp`. Demak parent-growth retsepti
va sparse active path (~`1.79M / 25.79M`, `6.95%`) tasdiqlandi, lekin bu gate’da
qo‘shimcha 200M virtual capacity sifatga qo‘shimcha bermadi. 1B ga faqat hajm
uchun o‘tish rad qilindi; P-003 direct-v0 va unseen-value muammolari ochiq.
**Batafsil:** `results/P003_TYPED_REGISTER_700M_SEED18_VALIDATION_20260907.md`.

### C-P003-TYPED-700M-OOD-001 — 700M unseen-value range audit

**Status:** `REJECTED AS A CAPACITY-ONLY GENERALIZATION FIX`
**Muammo:** P-003
**Natija:** Seed17 700M parent-growth modeli `0–31`da o‘qitilib, unseen
`32–63` qat’iy `32^3 x 9` gridda tekshirildi. Train-range `99.84%`, unseen
range `27.74%` bo‘ldi. Bu 500M growthdagi `28.57%`dan `−0.83 pp` va 300Mdagi
`27.35%`dan faqat `+0.39 pp`. Total `25.79M`, active estimate `1.79M`
o‘zgarmadi. Demak sig‘imni 700Mga oshirish unseen generalizationni hal qilmadi;
1B capacity-only run rad qilindi. Keyingi yo‘l representation yoki teacher-
derived activation transfer.
**Batafsil:** `results/P003_TYPED_REGISTER_700M_OOD_RANGE_20260907.md`.

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

Inference-only correction-gain sweep ham bajarildi: 100M staged seed17/18
checkpointlarda `circuit_delta_scale={0.5,1,2,4}` tekshirildi. `scale=4`
seed18 natural accuracy’ni `1.0`ga nisbatan `−1.88 pp` tushirdi; global va
within-task route replay CE/accuracy ta’siri seedlar orasida qarama-qarshi
bo‘ldi. Correction amplitudasini oshirish barqaror causal signal bermadi va
training/default uchun rad qilindi.

**Sweep audit:** `results/P007_CORRECTION_GAIN_SWEEP_AUDIT.md`.
