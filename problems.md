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

## Track chegarasi

P-001–P-007 va quyidagi tarixiy C-* yozuvlari Native Engine tadqiqot oqimiga
tegishli. Sparse Qwen va umumiy runtime muammolari alohida yuritiladi:
`RESEARCH_TRACKS.md` dagi `QWEN-001` va `RUNTIME-001`. Natijalarni bir-biriga
aralashtirmaslik kerak, chunki benchmark va teacher/reference ta’riflari
farq qiladi.

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

2026-09-08 candidate-window screen’da 100M/300M/500M seed17/18 checkpointlari
1,920 ta bir xil held-out misolda `candidate_pool=32,64,128,256` bilan
inference-only tekshirildi. O‘rtacha delta 32-poolga nisbatan 64/128/256 uchun
mos ravishda `+0.000889/+0.000306/−0.000077 CE` va
`−0.095/−0.009/−0.026 pp accuracy` bo‘ldi. Seedlar qarama-qarshi yo‘nalishda
ketdi; 500M seed17’dagi `256` pool yaxshilanishi seed18’da takrorlanmadi.
Demak oddiy lokal oynani kengaytirish katta muammoni hal qilmaydi va default
candidate pool o‘zgartirilmadi. Bu test learned retrievalni butunlay inkor
qilmaydi, lekin P-001ning “32 pool torligi asosiy sabab” qismini zaiflashtiradi.

**Audit:** `results/P001_CANDIDATE_POOL_WIDTH_SCREEN_20260908.md`.

**Route-neighborhood audit (2026-09-08):** 100M/300M/500M seed17/18 frozen
checkpointlarda natural route uchun bir circuitlik one-swap qidiruv qilindi.
Candidate pool ichidagi local selection headroom `0.00357–0.00506 CE` bo‘ldi.
Full-bank key top-8 circuitlarini target CE bilan diagnostik tekshirganda local
probe ustiga qo‘shimcha `0.00558–0.02199 CE` opportunity chiqdi; full-key top-8
candidate recall esa faqat `0.10–0.97%` edi. 500M’da retrieval opportunity
local selection headroomidan `3.3–5.8x` katta.

Bu full-bank key route productionga tayyor degani emas: key score target costga
mos emasligi sabab oldingi full-bank screen natural qualityni oshirmagan. Ammo
capacity kattalashganda useful circuitlar bor-u, current key/tree retriever
ularni ko‘rmayotgani va candidate ichidagi selector ham mukammal emasligi
aniqroq ko‘rindi. P-001 uchun keyingi yo‘l hidden state + circuit output
signature’dan final-cost surrogate; oddiy pool enlargement va full-bank key
scoring `REJECTED AS DIRECT FIX`.

**Audit:** `results/P001_ROUTE_NEIGHBORHOOD_REGRET_AUDIT_20260908.md`.

**Query/key cost-surrogate screen (2026-09-08):** 20M seed17/18da train
splitdan `92,160` one-swap CE label bilan kichik `261,705`-parametrli MLP
o‘qitildi. Feature’lar query, candidate key, selected-route summary va
key-score edi. Held-out surrogate route delta’si seed17/18da mos ravishda
`−0.000359/+0.000616 CE`, ikki-seed o‘rtachasi `+0.000129 CE` bo‘ldi; oracle
gain recovery `−3.75%/6.87%`, top-1 match `3.47%/2.92%`. Calibration loss
pasaygan bo‘lsa ham final route tanlovi generalizatsiya qilmadi.

Demak target-cost router g‘oyasi hozirgi feature set bilan ishlamadi. Muammo
candidate circuitning query-dependent output signalini ko‘rmayotgan bo‘lishi
mumkin; output signature qo‘shish esa active budget va routing costni yashirin
oshirmasligi kerak. Shu sabab full modelga integratsiya qilinmadi.

**Audit:** `results/P001_ROUTE_COST_SURROGATE_AUDIT_20260908.md`.

**Cost-router ketma-ket screenlari (2026-09-08):** Query/key MLP held-outda
ikki seed o‘rtachasida atigi `+0.000129 CE` berdi. 8/32/64-D candidate output
signature variantlarining o‘rtacha delta’si mos ravishda `+0.000009/+0.000407/
+0.000585 CE` bo‘ldi; 64-D signature routing yo‘lini `3.16x` qimmatlashtirdi.
Faqat `router.keys`ni one-swap CE label bilan target-align qilish esa
agressiv va konservativ LR/anchor nazoratlarida ham ikki seedda regressiya
berdi (`+0.00283…+0.00572 CE`).

Shu sabab oddiy cost MLP, output sketch va post-hoc key retraining direct fix
sifatida rad qilindi. P-001 ochiq: oracle retrieval headroom mavjud, lekin uni
active budgetni saqlagan holda end-to-end trainable utilityga aylantirish
kerak.

**Jamlangan audit:** `results/P001_COST_ROUTER_AUDITS_20260908.md`.

**Final output-logit cost feature screen (2026-09-10):** Candidate circuit
outputlari mavjud output head orqali final logitsga proyeksiya qilinib, frozen
20M seed17/18 banklarda one-swap final CE label bilan 1,000 qadamli surrogate
o‘qitildi. Held-out predicted-route CE delta’si `−0.000328/−0.000470` bo‘ldi;
oracle recovery `−3.91%/−6.58%`, top-1 oracle match `3.33%/2.64%`. Calibration
loss past bo‘lsa ham ikkala seedda predicted route natural route’dan yomonroq
chiqdi. Bu candidate output informationning o‘zi foydasizligini emas, frozen
post-hoc MLP va 32-candidate feature hisoblash direct fix emasligini bildiradi.
Variant `REJECTED FOR ADOPTION`; P-001 ochiq qoladi.

**Audit:** `results/P001_OUTPUT_LOGIT_COST_ROUTER_AUDIT_20260910.md`.

**Final-CE target retrieval distillation (2026-09-10):** Frozen 20M
seed17/18da one-swap final CE bilan eng yaxshi local yoki full-key-top-8
alternative circuit group target qilinib, faqat hierarchical tree/key router
1000 qadam o‘qitildi. Teacher targetlarning `70.69%/71.04%`i current
candidate pooldan tashqarida va target headroom `0.018019/0.015981 CE` edi.
Shunga qaramay held-out treatment CE `+0.020154/+0.014433`, accuracy
`−0.42/−2.08 pp` yomonlashdi. Offline frozen-query target distillation
`REJECTED FOR ADOPTION`; P-001 ochiq qoladi. Keyingi variant on-policy yoki
joint end-to-end retrieval bo‘lishi kerak, 300M/500Mga scale qilinmaydi.

**Audit:** `results/P001_TARGET_RETRIEVAL_DISTILL_AUDIT_20260910.md`.

**On-policy target aggregation (2026-09-10):** Frozen-body retrieval training
har rounddan keyin yangi natural query/state’dan one-swap final-CE targetlarni
qayta yig‘ib, 3 round davom ettirildi. Seed17/18 treatment CE delta’si
`+0.019016/+0.011590`, accuracy delta’si `−0.42/−2.50 pp` bo‘ldi; ikki-seed
mean `+0.015303 CE`, `−1.46 pp`. Stale calibration yagona sabab emasligi
ko‘rindi. Offline va on-policy target distillation oilasi `REJECTED FOR
ADOPTION`; P-001 ochiq, keyingi yo‘l body+router joint composition/state
interface bo‘ladi.

**Audit:** `results/P001_ON_POLICY_TARGET_RETRIEVAL_AUDIT_20260910.md`.

**Nonlinear candidate-score scorer (2026-09-08):** zero-initialized
`2*state_dim → 32 → 1` residual scorer bilan 20M seed17/18da 2,000-step
soft-routing continuation qilindi. Treatment CE delta’si `+0.001152/+0.020205`,
accuracy delta’si `−0.208/−0.573 pp` bo‘ldi. Ikkala seedda ham regressiya;
`24,641` qo‘shimcha scorer parametri va trainingda 32 candidate circuitning
soft aktivatsiyasi quality foydasini bermadi. Opt-in patch `REJECTED`.

**Audit:** `results/P001_NONLINEAR_ROUTE_SCORER_AUDIT_20260908.md`.

**Route-weight ablation (2026-09-08):** selected ID’lar va route gain’lar
muzlatilgan holda 100M/300M/500M seed17/18da faqat `K=8` weight’lari
almashtirildi. Uniform weight natural route’ga nisbatan barcha 6 run’da CE’ni
yaxshiladi, o‘rtacha delta `−0.000201`, ammo accuracy delta `−0.009 pp` bo‘ldi.
Power-half `−0.000108 CE`, power-2 `+0.000273 CE`, top-1 esa `+0.021070 CE`
va `−0.391 pp` berdi. Soft mixing muhim, lekin uniform/flattened variantning
foydasi juda kichik va selected circuitlarni o‘zgartirmaydi. `REJECTED AS A
PRIMARY FIX`; default o‘zgarmadi.

**Audit:** `results/P001_ROUTE_WEIGHT_ABLATION_AUDIT_20260908.md`.

**Uniform-weight continuation (2026-09-08):** 20M seed17/18da natural
query-key top-8 weight bilan control va uniform `1/K` weight bilan treatment
2,000 continuation step yurdi. Uniform deployment seed17’da `−0.002880 CE,
+1.094 pp`, seed18’da `+0.002210 CE, −0.104 pp` berdi; o‘rtacha
`−0.000335 CE, +0.495 pp`. Treatmentni natural weight bilan eval qilganda ham
o‘rtacha `−0.000247 CE, +0.521 pp` bo‘ldi. Seedlar orasida barqarorlik va `+2
pp` gate yo‘q; weighting primary fix sifatida `REJECTED FOR ADOPTION`, natural
default saqlandi.

**Audit:** `results/P001_UNIFORM_WEIGHT_CONTINUATION_AUDIT_20260908.md`.

**Route-neighborhood audit (2026-09-08):** 100M/300M/500M seed17/18 frozen
checkpointlarda natural route uchun bir circuitlik one-swap qidiruv qilindi.
Candidate pool ichidagi local selection headroom `0.00357–0.00506 CE` bo‘ldi.
Full-bank key top-8 circuitlarini target CE bilan diagnostik tekshirganda local
probe ustiga qo‘shimcha `0.00558–0.02199 CE` opportunity chiqdi; full-key top-8
candidate recall esa faqat `0.10–0.97%` edi. 500M’da retrieval opportunity
local selection headroomidan `3.3–5.8x` katta.

Bu full-bank key route productionga tayyor degani emas: key score target costga
mos emasligi sabab oldingi full-bank screen natural qualityni oshirmagan. Ammo
capacity kattalashganda useful circuitlar bor-u, current key/tree retriever
ularni ko‘rmayotgani va candidate ichidagi selector ham mukammal emasligi
aniqroq ko‘rindi. P-001 uchun keyingi yo‘l hidden state + circuit output
signature’dan final-cost surrogate; oddiy pool enlargement va full-bank key
scoring `REJECTED AS DIRECT FIX`.

**Audit:** `results/P001_ROUTE_NEIGHBORHOOD_REGRET_AUDIT_20260908.md`.

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

P-008 task-reuse/path-variance regularizer allaqachon sinovdan o‘tdi va
`REJECTED FOR ADOPTION` qilindi; uni takrorlamaymiz. Keyingi ish avval 100M va
300M staged checkpointlarda route causal ta’sirini circuit contribution,
candidate inclusion va recurrent-state bypass bo‘yicha alohida ajratadigan
diagnostic bo‘ladi. Faqat qaysi bo‘g‘in sabab ekani ko‘rsatilgandan keyin bitta
minimal opt-in patch sinov qilinadi. Active budget yoki route’ni majburan bir
xil qilish qabul qilinmaydi.

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

**Task-wise frontier reanalysis (2026-09-08):** oldingi staged 10k all-screen
natijalari task bo‘yicha ajratilganda 100M/300M/500Mda depth-1 mean
`99.67/99.83/99.61%`, depth-2 `72.14/71.68/72.14%`, depth-3 esa
`53.26/53.78/53.71%` bo‘ldi. `chain3` o‘rtacha `20.64%`, `state_machine`
`8.41%`, `compose_add_mul` `60.55%`da qolgan; capacity oshishi bu difficult
composition tasksni siljitmagan. Demak keyingi muammo raw capacity emas,
intermediate state/dataflow va composition circuit trainingidir.

**Audit:** `results/P003_TASKWISE_CAPACITY_FRONTIER_AUDIT_20260908.md`.

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

**Full-range output codec screen (2026-09-11):** range `0--63` depth-4
uchun mustaqil uchta digit head va Fourier-base alignment tekshirildi. Base
`1024` aligned codec ikki seedda `67.97%` mean held-out accuracy va `58.40%`
depth-4 berdi. Keyingi yumshoq cross-digit context variantida oldingi digit
taqsimoti embedding orqali keyingi headga uzatildi; natija `68.85%` mean va
`59.77%` depth-4 bo‘ldi, lekin mean CE `+0.0691` yomonlashdi va seed18
depth-4 o‘zgarmadi. Qo‘shimcha `73,728` parametr bo‘lsa ham `+2 pp` gate
 bajarilmadi. Variant `REJECTED FOR DEFAULT`, opt-in diagnostika sifatida
qoldi. Bu full-range codec muammosi capacity-only emasligini kuchaytiradi:
keyingi yo‘l digit headsni kattalashtirish emas, structured carry/quotient
contract yoki two-digit codecning range chegarasini to‘g‘ri kengaytirish
bo‘lishi kerak.

**Audit:** `results/V0_215_DYNAMIC_NONMOD_CROSS_DIGIT_INTERACTION_AUDIT.md`.

**Representation granularity screen (2026-09-11):** A two-digit base-32768
codec failed to learn the train split (`51.66%` mean) and reached only
`11.72%` held-out / `8.79%` depth-4. Keeping each local decision small with a
four-digit base-512 codec instead reached `79.59%` held-out and `74.41%`
depth-4 across seed17/18, while using `7.47M` total and `2.17M` estimated
active parameters. Fresh seed19 reaches `74.41%` overall and `60.55%` depth-4,
while seed20 reaches `78.52%`/`71.48%`; the four-seed mean is
`78.03%`/`70.21%`. The seed variance is material but the mean remains
`+10.06/+11.81 pp` above the aligned three-digit control. This is a strong
positive signal that the full-range failure is partly an output-code
granularity/interface problem, not simply insufficient bank capacity. The
four-digit path is now the leading full-range opt-in and remains out of
default pending a longer matched run; the two-digit path is `REJECTED`.

**Audits:** `results/V0_216_DYNAMIC_NONMOD_TWO_DIGIT_BASE32768_AUDIT.md`,
`results/V0_217_DYNAMIC_NONMOD_FOUR_DIGIT_BASE512_AUDIT.md`.

**Longer-budget verification (2026-09-11):** Fresh 5000-step runs on all four
seeds with the four-digit base-512 codec reach `81.20%` mean held-out and
`74.61%` depth-4, versus `78.03%`/`70.21%` for their matched 3000-step runs.
Seed19 improves especially strongly (`60.55% → 70.70%` at depth 4), so some
earlier seed variance was insufficient optimization budget. The output-code
granularity hypothesis is strengthened and this is now the leading full-range
candidate; default adoption waits for target-offset and fresh-data robustness.

**Audit:** `results/V0_218_DYNAMIC_NONMOD_FOUR_DIGIT_BASE512_5000STEP_AUDIT.md`.

**Offset robustness (2026-09-11):** Changing the target offset from `1,048,576`
to `2,097,152` on matched seed17/18 four-digit runs changes mean held-out
accuracy only `79.59% → 79.30%` and depth-4 only `74.41% → 74.22%`. The
four-digit gain is not absolute-offset memorization. The remaining active
question is fresh value/operation-distribution robustness, not raw capacity.

**Audit:** `results/V0_219_DYNAMIC_NONMOD_FOUR_DIGIT_BASE512_OFFSET_ROBUSTNESS.md`.

**Unseen value-range screen (2026-09-11):** Training only on operands
`0..31` and evaluating on `32..63` gives `51.66%` mean accuracy and `47.85%`
depth-4 despite `99.22%` train accuracy. The four-digit representation is
strong for full-range training but does not solve value-range extrapolation;
P-003 remains active for an explicit algebraic value/carry contract or
curriculum. An initial unsafe-offset attempt was rejected by the target-range
guard and is not counted as a quality result.

**Audit:** `results/V0_220_DYNAMIC_NONMOD_FOUR_DIGIT_UNSEEN_VALUE_RANGE_AUDIT.md`.

**Fixed Fourier input-encoder screen (2026-09-11):** Replacing the learned
value encoder with the existing fixed mod-64 Fourier encoder on the same
`0..31 → 32..63` unseen-range gate gives `46.29%` mean accuracy and `41.99%`
depth-4, versus `51.66%` and `47.85%` for the learned-encoder control. Mean CE
worsens from `10.1647` to `10.6830`; train accuracy remains `99.32%`. Thus the
input embedding alone is not the direct fix. The fixed Fourier option is
`REJECTED AS A DIRECT OOD FIX`; P-003 remains active for a value/carry contract
that survives recurrent composition and final readout. Capacity-only 700M/1B
scaling is still deferred.

**Audit:** `results/V0_221_DYNAMIC_NONMOD_FIXED_FOURIER_UNSEEN_VALUE_RANGE_AUDIT.md`.

**Cross-digit output interaction screen (2026-09-11):** Existing rank-32
conditional digit context was enabled on the same `0..31 → 32..63` gate.
Two-seed mean unseen accuracy rose `51.66% → 55.96%` (`+4.30 pp`), depth-4
`47.85% → 50.59%` (`+2.73 pp`), and CE improved `10.1647 → 9.9377`. The cost
was `45,312` total/active-estimate parameters; router and circuit computation
were unchanged. This is the strongest current signal that carry/cross-digit
readout contributes to P-003, but 3000 steps are insufficient for adoption.
The variant is `RETAINED AS OPT-IN`; next gate is matched 5000-step unseen-range
continuation plus full-range `0..63` regression. Capacity-only scaling remains
deferred.

**Audit:** `results/V0_222_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_OOD_AUDIT.md`.

**5000-step continuation (2026-09-11):** Cross-digit interaction rank-32
reaches `60.16%` mean unseen-range accuracy and `55.86%` depth-4 after 5000
steps, up from `55.96%`/`50.59%` at 3000 steps. Mean CE nevertheless worsens
from `9.9377` to `12.0836`; this is not yet attributable to the interaction
because a matched no-interaction 5000-step control is missing. The signal is
`PROMISING OPT-IN`, not solved. Required next controls are matched 5000-step
no-interaction OOD and interaction full-range `0..63` regression; default and
capacity scaling remain unchanged.

**Audit:** `results/V0_223_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_OOD_5000_AUDIT.md`.

**Matched 5000-step control (2026-09-11):** Against a fresh no-interaction
control with the same split, seeds, and budget, output interaction rank-32
improves unseen accuracy by `+3.91 pp` mean and depth-4 by `+4.10 pp`; CE also
improves by `−0.3212`. Both seeds agree. The added cost is `45,312` total and
estimated active parameters, with router/circuit computation unchanged. This
validates the carry/cross-digit readout hypothesis as an opt-in direction, but
P-003 is not solved: full-range `0..63` regression and rank/cost ablation are
still required before default adoption.

**Audit:** `results/V0_224_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_MATCHED_5000_CONTROL.md`.

**Full-range regression (2026-09-11):** With operands `0..63` in both train
and evaluation, interaction rank-32 reaches `83.98%` mean held-out and
`79.49%` depth-4 versus the matched no-interaction `82.62%`/`77.54%`, a
`+1.37/+1.95 pp` hard-quality gain. CE regresses by `+0.2128`, especially on
seed18, so the variant passes hard quality but is not default. P-003 remains
active for generalization/calibration; next is rank-16 OOD cost ablation.

**Audit:** `results/V0_225_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_FULL_RANGE_5000.md`.

**Rank-16 cost ablation (2026-09-11):** On the same unseen-range gate,
rank-16 interaction beats the matched no-interaction control by `+2.64 pp`
overall and `+3.52 pp` depth-4; CE improves by `−0.5941` across seed17/18.
The added total/active-estimate budget is only `22,656`, about half of rank32.
This is a promising lower-cost opt-in, not default; rank8 OOD and full-range
regression for the best rank remain.

**Audit:** `results/V0_226_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_RANK16_OOD.md`.

**Rank-8 cost ablation (2026-09-11):** On the same unseen-range gate, rank-8
interaction beats no interaction by `+2.54 pp` mean overall and `+2.93 pp`
depth-4, with CE improving `−0.1341`. It adds only `11,328` total and
estimated active parameters; hard quality is within `0.10 pp` overall and
`0.59 pp` depth-4 of rank16. This is the current low-cost opt-in, pending
full-range regression and final rank selection.

**Audit:** `results/V0_227_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_RANK8_OOD.md`.

**Rank-8 full-range regression (2026-09-11):** The low-cost rank-8 path fails
the full-range `0..63` gate: mean held-out `80.37%` and depth-4 `75.00%` versus
no-interaction `82.62%`/`77.54%`, with CE worsening by `+0.1751`. It is
`REJECTED FOR DEFAULT` despite the unseen-range gain. Rank16 full-range
regression is the remaining cost/quality decision between rank8 and rank32.

**Audit:** `results/V0_228_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_RANK8_FULL_RANGE.md`.

**Rank-16 full-range regression (2026-09-11):** Rank16 interaction beats the
matched no-interaction control by `+1.27 pp` mean held-out and `+1.56 pp`
depth-4, with CE worsening only `+0.0432` and `22,656` extra total/active-
estimate parameters. It is nearly rank32's hard quality at roughly half the
cost, making rank16 the leading quality/cost opt-in candidate. Four-seed
full-range validation is still required before default adoption.

**Audit:** `results/V0_229_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_RANK16_FULL_RANGE.md`.

**Four-seed full-range validation (2026-09-11):** Rank16 interaction reaches
`83.40%` mean held-out and `77.05%` depth-4 across seeds17/18/19/20, versus
the existing four-seed no-interaction reference `81.20%`/`74.61%`, a
`+2.20/+2.44 pp` gain. CE is effectively unchanged (`2.6888` vs `2.6938`).
The rank16 path is now validated as the leading full-range quality/cost
opt-in; exact four-seed unseen-range validation remains before default.

**Audit:** `results/V0_230_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_RANK16_FULL_RANGE_4SEED.md`.

**Four-seed unseen-range validation (2026-09-11):** Rank16 interaction beats
the matched no-interaction control on `0..31 → 32..63` by `+2.78 pp` mean
overall and `+2.93 pp` depth-4; CE improves `−0.3815`. All four seeds are
positive. Rank16 is now the leading quality/active-budget opt-in candidate;
P-003 remains active for above-range generalization and default validation.

**Audit:** `results/V0_231_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_RANK16_OOD_4SEED.md`.

**Above-range extrapolation planned (2026-09-11):** The next P-003 gate trains
rank16 on operands `0..63` and evaluates on `64..95`, with a larger safe target
offset. A matched no-interaction control is required; this is a representation
generalization test and does not authorize 700M/1B capacity scaling.

**Audit/config:** `results/V0_232_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_RANK16_OOD_ABOVE_RANGE.md`.

**Above-range screen invalidated (2026-09-11):** The attempted `0..63 →
64..95` run exposed a benchmark/input defect: non-modular `encode_tokens`
still used an effective 64-value range, so the held-out operands were not
distinct at the model input. Its completed metrics are excluded from all
architecture decisions. The fix is an explicit `value_encoder_modulus=128`
parameter and a corrected paired rerun in V0.233.

**Audit:** `results/V0_232_DYNAMIC_NONMOD_FOUR_DIGIT_CROSS_DIGIT_INTERACTION_RANK16_OOD_ABOVE_RANGE.md`.

**Corrected rerun (four seeds):** With `value_encoder_modulus=128`, rank16
improves hard accuracy by `+0.98 pp` overall and `+0.98 pp` depth-4 on the
valid `0..63 → 64..95` screen; the gain is non-negative in all four seeds but
small, and CE worsens by `+0.3885`. Keep it opt-in; this does not justify
700M/1B scaling or a default change.

**Corrected rerun:** `results/V0_233_DYNAMIC_NONMOD_VALUE_ENCODER_RANGE_FIX_ABOVE_RANGE.md`.

**Hybrid Fourier follow-up (2026-09-11):** Adding the fixed Fourier basis to
the corrected value128 encoder did not improve the interaction path: the
two-seed hybrid interaction mean was `58.40%` overall / `52.34%` depth-4
versus hybrid control `58.79%`/`52.15%`, with CE `+1.1784` worse. It is
rejected for adoption; learned-value128 rank16 remains the hard-quality
opt-in candidate and P-003 stays active.

**Audit:** `results/V0_234_DYNAMIC_NONMOD_HYBRID_VALUE_ENCODER_ABOVE_RANGE.md`.

**Straight-through hard digit context (2026-09-11):** Replacing the soft
previous-digit distribution with a straight-through hard argmax in the
rank16 cross-digit interaction gives a four-seed gain on the corrected
above-range screen: held-out `57.13% → 58.89%` (`+1.76 pp`), depth-3
`64.84% → 67.09%` (`+2.25 pp`), and depth-4 `49.41% → 50.68%`
(`+1.27 pp`). Mean CE improves by `−0.2593`; overall accuracy is positive in
all four seeds, although seed17 depth-4 falls `0.78 pp`. **VALIDATED AS
OPT-IN; DEFAULT UNCHANGED.** The signal remains below the `+2 pp` default
gate, is not evidence for 700M/1B scaling, and does not close P-003. The
next test should validate task/fresh-distribution robustness or move to a
distinct state/value-contract hypothesis rather than increasing capacity.

**Audit:** `results/V0_235_DYNAMIC_NONMOD_STRAIGHT_THROUGH_HARD_DIGIT_CONTEXT.md`.

**Task-wise hard-context diagnosis (2026-09-11):** The four-seed operation
breakdown shows that the V0.235 gain is localized: `add` is `100%` in both
arms, hard-context raises `subtract` from `57.52%` to `78.81%` overall and
from `27.73%` to `59.67%` at depth-4, but `multiply` is `0%` in both arms at
all held-out depths. Multiply CE is also slightly worse under hard context.
Therefore the remaining hard ceiling is not a generic router failure and
hard digit context is not the solution to it. **RETAIN HARD-CONTEXT AS
OPT-IN ONLY; KEEP DEFAULT UNCHANGED.** The next architectural experiment
should isolate an operation-specific multiply state transition/dataflow,
with add/subtract controls; no 700M/1B scaling follows yet.

**Audit:** `results/V0_236_DYNAMIC_HARDCONTEXT_TASKWISE_AUDIT.md`.

**Operation-conditioned output readout (2026-09-11):** A per-operation
rank-16 output adapter added `38,016` parameters but changed aggregate
held-out accuracy only `+0.49 pp` and worsened CE by `+1.5459`. Task-wise
multiply stayed at `0%` for both control and treatment at depths 3/4; only
subtract moved slightly. **REJECTED FOR ADOPTION.** The issue is not merely
missing operation identity at the final readout. The next test must pass the
existing exact algebraic value packet directly into the output codec, with
the learned state and circuit path unchanged.

**Audit:** `results/V0_237_DYNAMIC_NONMOD_OPERATION_OUTPUT_ADAPTER_AUDIT.md`.

**Direct algebraic output bridge (2026-09-11):** Reusing the existing exact
polynomial2/Fourier packet after output LayerNorm adds no parameters, but the
matched task-wise multiply result remains `0%` at depths 3/4. Subtract alone
improves; aggregate CE worsens by `+0.7328`. **REJECTED FOR ADOPTION.** Before
adding another state mechanism, test whether the failure is simply product-
range OOD: train on operands `0..95` and evaluate unseen depths on the same
range. If multiply then works, the prior `0..63 → 64..95` result is a data
support limitation, not evidence for more capacity.

**Audit:** `results/V0_238_DYNAMIC_NONMOD_ALGEBRAIC_OUTPUT_BRIDGE_AUDIT.md`.

**Wide-support control (2026-09-11):** Training on operands `0..95` while
holding out only depths restores non-zero multiply accuracy without changing
capacity: mean held-out multiply `15.43%`, depth-3 `23.24%`, depth-4 `7.62%`;
add is `100%` and subtract `91.99%`. The earlier `0..63 → 64..95` multiply
`0%` result was therefore partly a product-range OOD confounder. The clean
protocol is now wide-support/depth-holdout for architecture comparisons.
P-003 remains active because deep multiply is still weak; **no 700M/1B
scaling** follows. Next work should improve deep multiply dataflow on this
clean protocol, not add raw capacity.

**Audit:** `results/V0_239_DYNAMIC_NONMOD_WIDE_TRAIN_RANGE_CONTROL.md`.

**Clean wide-support hard-context follow-up (2026-09-11):** On the matched
`0..95` operand/depth-holdout protocol, straight-through hard digit context
raises aggregate held-out accuracy only `79.199% → 79.688%` (`+0.488 pp`)
and improves CE `3.554884 → 3.312335`. The operation-wise result is decisive:
`add` stays `100%`, `subtract` rises `91.99% → 99.71%`, but `multiply` is
unchanged at `15.43%`; depth-4 multiply is only `7.62% → 7.81%`. Therefore
hard context is **RETAINED AS OPT-IN FOR SUBTRACT, DEFAULT UNCHANGED**, while
the main P-003 ceiling remains an operation-specific multiply state/dataflow
transition. Another router/capacity sweep is deferred; the next test should
modify that transition under the same clean protocol.

**Audit:** `results/V0_240_DYNAMIC_NONMOD_WIDE_RANGE_HARDCONTEXT_AUDIT.md`.

**Algebraic write-bridge follow-up (2026-09-11):** Reusing the exact
polynomial/Fourier packet before the learned state writer gives a small
aggregate taskwise gain, `82.715% → 83.301%` (`+0.586 pp`), but it worsens
subtract `99.707% → 97.461%` and leaves multiply unchanged at `15.430%`;
depth-4 multiply remains `7.8125%`. **REJECTED FOR MAIN QUALITY ADOPTION.**
The packet is not enough as an additive write hint. P-003 remains active; the
next experiment will test authoritative algebraic read dataflow with the
same circuits, budget, and clean wide-support protocol.

**Audit:** `results/V0_241_DYNAMIC_NONMOD_ALGEBRAIC_WRITE_BRIDGE_AUDIT.md`.

**Algebraic authoritative-read follow-up (2026-09-11):** Replacing the
learned accumulator with the exact algebraic packet projection for pair/router
reads lowers aggregate taskwise accuracy `82.715% → 82.227%` and depth-4
`78.906% → 77.734%`; subtract falls `99.707% → 95.605%`. Multiply remains
unchanged at `15.430%` and depth-4 at `7.8125%`. **REJECTED FOR MAIN QUALITY
ADOPTION.** P-003 is therefore not solved by choosing a different query state
source. The next test should isolate a learned packet-to-digit decoder before
any capacity scaling.

**Audit:** `results/V0_242_DYNAMIC_NONMOD_ALGEBRAIC_AUTHORITATIVE_READ_AUDIT.md`.

**Direct packet output decoder (2026-09-11):** A separate learned decoder
from the existing polynomial2/Fourier packet improves aggregate taskwise
accuracy `82.715% → 85.156%`, but only because subtract reaches `100%`;
multiply falls `15.430% → 13.574%` and depth-4 falls `7.8125% → 6.8359%`.
**REJECTED FOR MULTIPLY QUALITY ADOPTION.** This points to low-order precision
loss or aliasing in the normalized float packet for large products. The next
test uses a separate exact-integer packet with learned base-512 digit decoding;
no circuit-bank or raw-capacity increase.

**Audit:** `results/V0_243_DYNAMIC_NONMOD_ALGEBRAIC_OUTPUT_DECODER_AUDIT.md`.

**Exact integer output codec (2026-09-11):** A separate int64 algebraic
register plus learned base-512 digit embeddings raises multiply from
`15.430%` to `23.047%` mean held-out, depth-3 from `23.242%` to `36.719%`,
and depth-4 from `7.617%` to `9.375%`. However, using the codec for every
operation damages subtract `99.707% → 80.176%`. **RETAINED AS A STRONG
MULTIPLY-SPECIFIC DIAGNOSTIC, NOT DEFAULT.** This confirms numeric precision /
codec representation is part of P-003, while mixed operations need
operation-conditioned readouts. Next test: integer codec for multiply only;
learned readout for add/subtract.

**Audit:** `results/V0_244_DYNAMIC_NONMOD_EXACT_INTEGER_OUTPUT_CODEC_AUDIT.md`.

**Multiply-only integer decoder (2026-09-11):** Selecting the exact integer
decoder only for terminal multiply preserves a multiply gain (`15.430% →
22.559%`, depth-3 `23.047% → 35.938%`) but, when the whole model is retrained,
subtract collapses `99.707% → 79.688%` and aggregate accuracy falls to
`66.309%`. **REJECTED AS END-TO-END TRAINING CONFIGURATION.** The exact integer
signal remains valid; next step is a frozen V0.240 checkpoint overlay training
only the new multiply decoder so existing add/subtract cannot regress.

**Audit:** `results/V0_245_DYNAMIC_NONMOD_MULTIPLY_ONLY_INTEGER_CODEC_AUDIT.md`.

**Frozen integer overlay (2026-09-11):** Loading V0.240 and freezing the full
body while training only an exact-integer decoder plus a multiply-specific
digit head preserved add (`100.000%`) and subtract (`99.707%`) exactly, while
raising multiply from `15.430%` to `23.145%` (`+7.715 pp`) and aggregate
held-out accuracy from `82.715%` to `83.691%` across seed17/18. **RETAINED AS
STRONG OPT-IN SIGNAL; NOT DEFAULT.** The improvement is operation-specific and
adds `337,474` trainable parameters, so the next problem is reducing that
overlay budget without losing the gain.

**Audit:** `results/V0_246_DYNAMIC_NONMOD_FROZEN_INTEGER_OVERLAY_AUDIT.md`.

**Lower-rank frozen integer overlay (2026-09-11):** Rank64 overlay bilan
multiply `15.430% → 19.922%` (`+4.492 pp`), add `100.000%` va subtract
`99.707%` saqlanib qoldi. Bu rank128dagi `+7.715 pp` foydaning bir qismini
saqlaydi, lekin aggregate foyda faqat `+0.098 pp`; **RANK64 LOWER-BUDGET
DIAGNOSTIC SIFATIDA QOLDIRILDI, LEADING VARIANT EMAS.** Keyingi screen rank32.

**Audit:** `results/V0_247_DYNAMIC_NONMOD_FROZEN_INTEGER_OVERLAY_RANK64_AUDIT.md`.

**Rank32 frozen integer overlay (2026-09-11):** Overlay rankini 32ga tushirish
qualityni saqlamadi: aggregate `82.715% → 78.418%`, multiply `15.430% →
13.672%`; add/subtract muzlatilgani uchun `100.000%/99.707%` o‘zgarmadi.
**REJECTED FOR QUALITY.** Rank128 quality varianti bo‘lib qoldi; rank64 faqat
budget diagnostikasi.

**Audit:** `results/V0_248_DYNAMIC_NONMOD_FROZEN_INTEGER_OVERLAY_RANK32_AUDIT.md`.

**Integer codec calibration sweep (2026-09-11):** Exact output codecga
`−81.45M…+81.45M` random raw-value calibration qo‘shilganda weight1.0 va
0.25 aggregate qualityni tushirdi, ammo deep multiply yaxshilandi. Weight0.10
aggregate `82.715% → 82.813%`, multiply `15.430% → 52.930%`; weight0.05 esa
aggregate `82.715% → 84.180%`, multiply `15.430% → 59.863%`, depth-3
`80.078%`, depth-4 `39.648%` berdi. Add/subtract ikkala seedda saqlandi.
**V0.252 RETAINED AS LEADING OPT-IN SIGNAL; DEFAULT O‘ZGARMADI.** Bu output
codec va range-coverage foydasini ko‘rsatadi, lekin recurrent intermediate
dataflow va universal scale muammosini hali isbotlamaydi. Keyingi test exact
packetni learned query/router statega kichik scale bilan aralashtirishdir.

**Audit:** `results/V0_249_252_DYNAMIC_NONMOD_INTEGER_CODEC_CALIBRATION_SWEEP_AUDIT.md`.

**Exact packet recurrent query-read probe (2026-09-11):** V0.252 exact
integer packetni shared recurrent queryga fixed scale bilan qo‘shish sinovdan
o‘tkazildi. `scale=0.0` controlda aggregate `85.156%`, `0.5` da `81.348%`,
`1.0` da `69.336%`, `2.0` da `49.023%` bo‘ldi; `4.0` da `40.918%` gacha
tushdi. Multiply barcha scale'larda aynan `61.430%` bo‘lib qoldi, ya’ni
packet multiply yo‘liga kirmadi; salbiy ta’sir shared add/subtract pathga
bo‘ldi. **SHARED RAW QUERY INJECTION RAD ETILDI.** V0.252 terminal
multiply-only overlay sifatida qoladi; P-003 hal bo‘lgani yo‘q. Keyingi
sinov alohida operation-conditioned, normalized transition/gate bo‘lishi
kerak, umumiy query residuali takrorlanmaydi.

**Audit:** `results/V0_253_DYNAMIC_NONMOD_INTEGER_STATE_QUERY_PROBE_AUDIT.md`.

**All-operation frozen integer overlay (2026-09-11):** V0.254 exact integer
headni add/subtract/multiply terminal output uchun qo‘lladi. Body, router va
circuit bank muzlatilgan holda aggregate `84.180% → 97.852%`, multiply
`59.863% → 71.875%` bo‘ldi; add/subtract ikkala seedda `100%` ga chiqdi.
**STRONG OPT-IN NUMERIC READOUT SIFATIDA QOLDIRILDI; DEFAULT O‘ZGARMADI.**
Bu router yoki raw capacity yechimi emas, terminal numeric readout signalidir.

**Audit:** `results/V0_254_DYNAMIC_NONMOD_ALL_OPERATION_INTEGER_OVERLAY_AUDIT.md`.

**Full-range integer codec calibration (2026-09-11):** V0.255 codec
kalibrovkasini eski `±81.45M` dan barcha legal class targetlariga kengaytirdi.
Matched `0..95` held-out’da aggregate `98.047%`, add/subtract/multiply uchalasi
`100%` bo‘ldi. Unseen fixed operand `96` da aggregate `95.313%`, multiply
`100%` bo‘ldi; V0.254 unseen multiply `0%` edi. **V0.255 LEADING OPT-IN
QUALITY CANDIDATE.** Biroq exact algebraic state benchmark operatsiyasini
oldindan hisoblaydi, shuning uchun bu 700M/1B universal scaling isboti emas va
default learned path hali o‘zgartirilmaydi.

**Audit:** `results/V0_255_DYNAMIC_NONMOD_FULL_RANGE_INTEGER_CODEC_AUDIT.md`.

**Rank64 full-range codec screen (2026-09-11):** Overlay budgetni
`337,474 → 207,362` tushirish uchun rank64 sinov qilindi. Seed17 matched
`0..95` aggregate `90.430%`, add `100%`, subtract `81.055%`, multiply
`92.773%`; unseen fixed `96` da subtract `50%`, multiply `100%` bo‘ldi.
**QUALITY GATE BAJARILMAGANI UCHUN RAD ETILDI.** Rank128 V0.255 leading
opt-in bo‘lib qoldi; rank32 shu gate ostida to‘liq screen qilinmaydi.

**Audit:** `results/V0_256_DYNAMIC_NONMOD_RANK64_FULL_RANGE_CODEC_AUDIT.md`.

**Rank64 long-training continuation (2026-09-11):** 5k-step rank64 screendagi
underfitni ajratish uchun 15k qadamga uzaytirildi. Seed17/18 o‘rtachasi matched
`0..95` all `96.973%`, add `100%`, subtract `99.219%`, multiply `99.414%`;
unseen fixed `96` da all `95.605%`, add/subtract/multiply `100%` bo‘ldi.
Rank64 `207,362` trainable parametr bilan ishlaydi, lekin rank128 V0.255’dan
aggregate taxminan `1.07 pp` past va ko‘proq training vaqt oladi. **LOWER-
BUDGET OPT-IN SIFATIDA QOLDIRILDI; PEAK QUALITY EMAS.** Rank32 shu frontier
ostida qayta sinovga qo‘yilmadi.

**Audit:** `results/V0_257_258_DYNAMIC_NONMOD_RANK64_LONG_TRAIN_AUDIT.md`.

**Depth-3 unseen value extrapolation (2026-09-11):** V0.255 `96..127`
unseen qiymatlarida, class overflow’ni chetlab depth-3 alohida tekshirilganda,
seed17/18 o‘rtachasi all `96.289%`, add/subtract/multiply esa `100%` bo‘ldi.
**VALUE-COVERAGE VALIDATSIYASI O‘TDI.** Bu codec range coverage’ning kuchli
dalili, lekin exact algebraic state known benchmark semanticsni beradi;
general learned arithmetic yoki 700M/1B scaling isboti emas.

**Audit:** `results/V0_259_DYNAMIC_NONMOD_DEPTH3_UNSEEN_VALUE_AUDIT.md`.

**Rank128 long-training codec continuation (2026-09-11):** V0.260/261 V0.255
overlay’iga yana 10k qadam qo‘shdi. Seed17/18 o‘rtachasi matched `0..95` all
`99.805%`, add/subtract `100%`, multiply `99.902%`; unseen fixed `96` da all
`96.680%`, add/subtract/multiply `100%` bo‘ldi. **PEAK OPT-IN CHECKPOINT
SIFATIDA QOLDIRILDI.** Bu natija body/router o‘zgarmagan holda chiqdi; paired
circuit ablation `0.0 pp` farq bergan, shuning uchun bu sparse routing yutug‘i
emas, numeric codec controlidir. Keyingi gate known exact algebraic priorni
kamaytirib learned circuit value-contractni tekshirish; 700M/1B scaling hozir
kerak emas.

**Audit:** `results/V0_260_261_DYNAMIC_NONMOD_RANK128_LONG_TRAIN_AUDIT.md`.

**Integer codec circuit ablation (2026-09-11):** V0.260/261 checkpointlarda
aynı held-out batch bilan circuit residual `1.0` va `0.0` solishtirildi.
Seed17/18’da all operations delta `0.0 pp`, add/subtract `0.0 pp`, multiply
ham `0.0 pp` bo‘ldi. **CODEC NATIJASIDA SPARSE CIRCUIT/ROUTER HISSASI
ANIQLANMADI.** Bu circuitlarni umumiy modeldan o‘chirish kerak degani emas;
faqat exact-prior opt-in branchdagi 99–100% raqamlar learned circuit yutug‘i
emas. Keyingi gate output learned state’ni bypass qila olmaydigan prior-free
task/dataflow benchmark bo‘lishi kerak.

**Audit:** `results/V0_262_DYNAMIC_NONMOD_INTEGER_CODEC_CIRCUIT_ABLATION_AUDIT.md`.

**Prior-dependence ablation (2026-09-11):** V0.263 was run on the peak
V0.260/261 checkpoints with three inference modes. Full exact codec quality
was `99.609–100.000%` on held-out `0..95`, while disabling only the exact
integer readout left the learned readout at `79.102–82.227%`; fixed unseen `96`
fell to `62.305–68.555%`. Disabling the algebraic state as well produced
`0.000%` in this inference-only stress test. Held-out learned multiply was
only `11.328–11.523%`, despite full-codec multiply at about `99.8%`. **The
peak score is therefore a numeric-prior/codec result, not a learned circuit
result.** V0.263 is diagnostic only; default unchanged.

**Audit:** `results/V0_263_264_DYNAMIC_NONMOD_PRIOR_ABLATION_AUDIT.md`.

**Retrained prior-free control (2026-09-11):** V0.264 removed both the exact
integer codec and algebraic state from the same 300M virtual body, then trained
two seeds for 5k steps on wide operands `0..95`, depths 1–2, with depths 3–4
held out. Train accuracy reached only `35.352%/41.602%`; held-out accuracy was
`6.836%/6.445%` (seed17/18). The operation probe gave held-out multiply
`5.273%/6.250%` and fixed unseen `96` all-operation `11.914%/16.406%`.
**V0.264 REJECTED.** This is a clean negative control: removing the prior and
retraining does not recover the composition dataflow, so the remaining P-003
ceiling is architectural state transition/circuit computation, not a missing
router screen or insufficient raw capacity. Next work is a learned typed
value contract with operation-specific transition capacity; no 700M/1B scale.

**Audit:** `results/V0_263_264_DYNAMIC_NONMOD_PRIOR_ABLATION_AUDIT.md`.

**Operation-conditioned bilinear write transition (2026-09-11):** V0.265
added a learned rank-16 accumulator×operand interaction at the state-write
boundary, with three operation-specific parameter sets and no exact numeric
prior. On the same prior-free wide-support/depth-holdout protocol, held-out
all accuracy moved `6.641% → 7.422%` across seed17/18 (`+0.781 pp`), below the
`+2 pp` adoption gate. The separate operation probe kept multiply near chance
(`5.664%/6.445%`) and fixed unseen-96 multiply at `0%/0%`. **RETAINED AS
OPT-IN DIAGNOSTIC, NOT ADOPTED.** The idea gives a small reproducible signal,
but does not establish a learned value contract or solve deep composition.
Next control: train prior-free variants on all depths to separate untrained
step extrapolation from a genuine transition failure; no rank increase or
700M/1B scale yet.

**Audit:** `results/V0_265_DYNAMIC_NONMOD_BILINEAR_TRANSITION_AUDIT.md`.

**Prior-free all-depth control (2026-09-11):** V0.266 trained the same
prior-free 300M-virtual model on depths 1–4, removing the possibility that
V0.264/265 failed only because depth-3/4 steps were untrained. Train accuracy
was just `12.012%/12.207%` and eval accuracy `11.719%/10.938%` across seed17/18;
depth-4 stayed `3.125%/5.469%`, and held-out operation probe multiply was
`6.445%/6.445%`. Routing used nonzero factor/virtual rows, so this is not
simply a dead-router artifact. **V0.266 REJECTED.** P-003 is now localized to
the learned value/state representation and its interface with sparse circuit
updates. Next test should add a small supervised learned value contract at
the recurrent boundary, retaining exact codec only as a separate hybrid
baseline; no 700M/1B scaling yet.

**Audit:** `results/V0_266_DYNAMIC_NONMOD_PRIOR_FREE_ALL_DEPTHS_AUDIT.md`.

**Bilinear all-depth control (2026-09-12):** V0.267 trained the rank-16
operation-conditioned accumulator×operand write residual on all depths 1–4,
removing the depth-holdout confound from V0.265. The small final evaluation
showed `11.328% → 13.574%` mean (`+2.246 pp`), but this was seed-driven
(`+0.195 pp` for seed17 versus `+4.297 pp` for seed18) and depth-4 did not
improve. A larger paired evaluation with 1,024 identical examples per depth
reduced the mean difference to only `11.902% → 12.134%` (`+0.232 pp`), with
seed17 regressing `12.134% → 9.912%`. Route diversity increased substantially
without a corresponding quality gain. **V0.267 REJECTED FOR ADOPTION AND
SCALING; OPT-IN DIAGNOSTIC ONLY.** P-003/P-004 remain active. The next test
must target a reusable typed value/carry contract rather than a larger
bilinear residual.

**Audit:** `results/V0_267_DYNAMIC_NONMOD_BILINEAR_ALL_DEPTHS_AUDIT.md`.

**Supervised scalar value-contract screen (2026-09-12):** V0.268 added the
existing one-dimensional `(old, operand, old*operand, bias)` scalar lane with
normalized intermediate-value supervision and query/read injection. On the
small `0..7` task with depths 1–2 train and 3–4 held out, the original reports
gave control/treatment means `67.773% → 54.883%` (`−12.891 pp`). A larger
paired held-out evaluation with 1,024 identical examples per depth gave
`68.921% → 56.494%` (`−12.427 pp`), with both seeds and both held-out depths
regressing. **V0.268 REJECTED.** This closes the scalar-contract family as a
quality fix; P-003/P-004 remain active, and the next representation must be a
genuinely vector-valued typed register/carry interface rather than more scalar
loss or scale sweeps.

**Audit:** `results/V0_268_DYNAMIC_NONMOD_STRUCTURED_CONTRACT_AUDIT.md`.

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

**Composition stage-signal audit (2026-09-08):** 100M/300M/500M staged 10k
checkpointlarda `step_logits` intermediate targetlar bilan tekshirildi.
`chain3` step-0 accuracy `2.99%`, step-1 `22.14%`, final `25.26%`; barcha
scale’da `state_machine` step-0/1 mos ravishda `2.34%/1.69%`, final `6.77%`
bo‘ldi. `reverse_sum` final `96.48%` bo‘lsa ham step-0 partial target `1.56%`
qolgan. Demak model ayrim tasklarda explicit intermediate registerni emas,
finalni bevosita taxmin qilmoqda; capacity qo‘shishdan oldin state transition
signalini alohida train qilish kerak.

**Audit:** `results/P004_COMPOSITION_STAGE_SIGNAL_AUDIT_20260908.md`.

**Keyingi opt-in test:** faqat depth-2/3 tasklar uchun `stage_loss_weight=0.1`
composition-only continuation; depth-1 auxiliary lossdan chiqariladi. Maqsad
oldingi umumiy stage supervisiondagi easy-task regressionni takrorlamasdan
intermediate state signalini ko‘tarish.

**Stage-only continuation natijasi (2026-09-08):** 20M matched checkpointlarda
2,000 qadamlik ikki-seed testda stage-0 accuracy `+6.927/+6.354 pp` ko‘tarildi,
ammo final accuracy `+0.729/−0.260 pp` bo‘ldi; mean foyda faqat `+0.234 pp`.
Mean CE `+0.008647` bilan yomonlashdi va ikkala seedda ham treatment CE’si
control’dan yuqori chiqdi. Shuning uchun composition-only auxiliary loss
`REJECTED FOR ADOPTION`; oddiy stage loss signalni kuchaytiradi, lekin
intermediate state’ni keyingi operation uchun foydali computationga aylantirmaydi.
P-004 `ACTIVE` qoladi. Keyingi yo‘l explicit typed intermediate register yoki
operation-conditioned state transition bridge; 700M/1B scaling bu dalil bilan
boshlanmaydi.

**Yakuniy audit:** `results/P004_COMPOSITION_STAGE_SIGNAL_AUDIT_20260908.md`.

**Typed-register bridge sinovi (2026-09-08):** `step_logits`ni 64-class typed
value embedding sifatida keyingi recurrent queryga qaytaruvchi opt-in bridge
2×2 nazorat bilan tekshirildi. Soft bridge + stage loss final accuracy’da
seed17/18 `+0.990/−0.313 pp`, mean `+0.339 pp`, mean CE `−0.006420` berdi;
stage-0 esa mean `+6.927 pp` ko‘tarildi, lekin stage-1/2 barqaror emas.
Straight-through bridge + stage loss `+0.208/−0.156 pp`, mean `+0.026 pp`,
mean CE `+0.008959` berdi. Ikkalasi ham `+2 pp` gate’dan o‘tmadi va
`REJECTED FOR ADOPTION` qilindi. P-004 ochiq qoladi: keyingi sinov umumiy
output reinjection emas, operation-conditioned typed transition bridge bo‘ladi.

**Batafsil:** `results/P004_TYPED_REGISTER_BRIDGE_AUDIT_20260908.md`.

**Operation-conditioned transition sinovi (2026-09-08):** state write’dan
oldin task-conditioned low-rank adapter tekshirildi. Rank-8 2k continuation
ikki seedda mean final accuracy `−0.026 pp`, mean CE `+0.003926` bo‘ldi.
Rank-32 2k’da mean `+0.703 pp` accuracy va `−0.012546` CE, 5k’da esa mean
`+0.312 pp` accuracy va `−0.023842` CE chiqdi; 5k depth-3 o‘rtacha accuracy
`−0.521 pp` bo‘ldi. CE foydasi takrorlangan bo‘lsa ham hard-quality gate
`+2 pp` bajarilmadi. Rank-32 transition `REJECTED FOR ADOPTION`, P-004
`ACTIVE` qoladi; keyingi yo‘l adapterni kattalashtirish emas, reusable
algebraic value/state primitive.

**Batafsil:** `results/P004_OPERATION_TRANSITION_AUDIT_20260908.md`.

**Route-conditioned circuit state-write adapter (2026-09-10):** Har bir
selected circuit `delta`si uchun zero-init rank-4 low-rank adapter qo‘shilib,
GRUCell update’iga route-specific write residual berildi. 20M seed17/18da
1000-step balanced continuation natijasi CE delta `−0.001533/+0.005050`,
accuracy delta `−0.391/+0.078 pp`; ikki-seed mean `+0.001759 CE`, `−0.156 pp`.
Active circuit params `101,376 → 125,952`, mean stats-free latency taxminan
`+2.5%`, peak VRAM `161 → 178 MB` bo‘ldi. Route identity’ni state write’ga
berish bu recipe’da composition qualityni barqaror oshirmadi; variant
`REJECTED FOR ADOPTION`, rankni oshirish yoki 300Mga scale qilish to‘xtatildi.

**Batafsil:** `results/P004_ROUTE_STATE_ADAPTER_AUDIT_20260910.md`.

**Multi-slot typed-register sinovi (2026-09-08):** oldingi bridge bitta
intermediate qiymatni overwrite qilgani sababli signal yo‘qolishi mumkin degan
gipoteza uchun step-0 va step-1 qiymatlarini alohida slotlarda saqlaydigan
`register_slot_count=2` varianti tekshirildi. Bridge-only final accuracy
seed17/18 `+0.365/−0.833 pp`, o‘rtacha `−0.234 pp` va mean CE `−0.001890`
bo‘ldi. Bridge+stage `+0.104/−1.250 pp`, o‘rtacha `−0.573 pp` va mean CE
`+0.005282` berdi. Stage-0 o‘rtacha `+6.667 pp` ko‘tarilgan bo‘lsa ham
stage-2 `−0.521 pp` tushdi. Demak faqat register tarixini saqlash yetarli
emas; muammo typed representation/state transition va circuit computation
moslashuvida ham bor. Multi-slot bridge `REJECTED FOR ADOPTION`, default
o‘zgarmadi va P-004 `ACTIVE` qoladi.

**Batafsil:** `results/P004_MULTI_SLOT_REGISTER_AUDIT_20260908.md`.

**Structured slot-read sinovi (2026-09-08):** ikki register slotini oddiy
sum emas, identity-initialized learned mixer bilan o‘qish tekshirildi.
Bridge-only final accuracy seed17/18 `−0.156/−0.990 pp`, o‘rtacha
`−0.573 pp`, mean CE `+0.011609` bo‘ldi. Bridge+stage `+0.313/−0.990 pp`,
o‘rtacha `−0.339 pp`, mean CE `+0.009105` berdi. Stage-0 o‘rtacha `+6.458 pp`
ko‘tarilgan bo‘lsa ham stage-2 faqat `+1.172 pp` bo‘ldi va
`state_machine` yaxshilanmadi; mixerning qo‘shimcha 319k
parametri ham foyda bermadi. Structured slot read `REJECTED FOR ADOPTION`,
default o‘zgarmadi va P-004 `ACTIVE` qoladi.

**Batafsil:** `results/P004_STRUCTURED_SLOT_MIXER_AUDIT_20260908.md`.

**State-only stage head sinovi (2026-09-08):** intermediate targetlarni final
output head’iga emas, recurrent state’dan o‘qiydigan alohida auxiliary head’ga
berish tekshirildi. Seed17/18 final accuracy delta `−0.052/−0.313 pp`, mean
`−0.182 pp`; mean CE `+0.008092` yomonlashdi. Stage-0 o‘rtacha `−0.286 pp`
bo‘lib, state representation foydali intermediate signalga aylanmadi.
`state_stage_head` `REJECTED FOR ADOPTION`, default o‘zgarmadi va P-004
`ACTIVE` qoladi.

**Batafsil:** `results/P004_STATE_STAGE_HEAD_AUDIT_20260908.md`.

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

---

### QWEN-001 — K=4 candidate retrieval va subset regret

**Status:** `ACTIVE`  
**Track:** Sparse Qwen  
**Prioritet:** P0

#### Dalil

- K=4 (`50%` active) learned routingda ikki seed CE delta `+0.06462` va
  `+0.06165` bo‘ldi.
- Xuddi shu circuitlarda exact paired-subset oracle `+0.01607` va `+0.01227`
  berdi.
- Demak foydali sparse subsetlar mavjud; joriy asosiy to‘siq learned
  candidate retrieval/subset tanlashdagi regret.

#### Qabul qilish mezonlari

- bir xil frozen children, K=4, sakkiz qatlam, seed17/18;
- candidate recall eski routerdan pasaymasin;
- p95 retrieval/selection regret kamida 10% yaxshilansin;
- hard accuracy va CE ikkala seedda yaxshilansin yoki gate’dan o‘tishi;
- dead circuits va sparse latency alohida hisoblansin.

#### Expert task

Faqat candidate retrieval yoki subset-selection signalini o‘zgartiradigan
minimal patch yozing. Circuit body, correction formulasi va recurrent state
update’ni bir vaqtda o‘zgartirmang. Exact paired oracle bilan gapni alohida
hisoblang; gate bajarilmasa `REJECTED` deb hujjatlashtiring.

---

### RUNTIME-001 — Small-batch va one-token decode overhead

**Status:** `ACTIVE`  
**Track:** Runtime  
**Prioritet:** P1

#### Dalil

- Qwen selected-token dispatch katta batchda deyarli dense darajasiga tushdi,
  ammo one-token decode K=5/K=6 da `1.371x/1.403x` bo‘lib qolmoqda.
- Native Engine V0.12 batch=128 da tez, lekin kichik batchda Python/PyTorch
  kernel-launch va routing overheadi hali asosiy xavf.

#### Qabul qilish mezonlari

- batch=128, batch=1 va sequence/one-token rejimlari alohida o‘lchansin;
- numerical output mavjud tolerance ichida aynan saqlansin;
- Native va Qwen natijalari bir jadvalda aralashtirilmasin;
- speedup bilan birga active bytes, kernel/dispatch va controller xarajati
  ko‘rsatilsin.

#### Expert task

Routing objective yoki model sifatini o‘zgartirmasdan selected-token dispatch,
memory layout yoki compiled/fused decode kernel uchun minimal patch yozing.
Avval profiler baseline, keyin benchmark va numerical equivalence testlarini
qo‘shing.

**2026-09-08 baseline:** Native 100M staged checkpoint RTX 3060’da
diagnostic-heavy path bilan batch-128 `10.871 ms`, batch-1 `4.881 ms` bo‘ldi;
dense reference mos ravishda `33.802 ms` va `3.296 ms`. `collect_stats=False`
serving path diagnostics tensorlarini yig‘masdan batch-128ni `9.618 ms`ga,
batch-1ni `3.737 ms`ga tushirdi va logitsni numerik teng saqladi. Keyingi
router metadata/entropy skip qayta o‘lchovda `8.077/3.572 ms` (batch-128/1)
berdi. Bu foydali
overhead patchi, lekin one-token latency muammosi yopilmadi: Native hali
dense’dan `1.084x`.

**Status update:** stats-free serving path `ACCEPTED FOR SERVING PATH`; compiled
decode/fused router kernel `ACTIVE`. Static CUDA Graph fixed-shape smoke logit
error `0.0`, lekin speed ratio faqat `0.959x/0.985x` (batch-1/128) bo‘ldi va
defaultga olinmadi. Float32 `matmul_precision=high` ham batch-1da atigi
`0.993x`, batch-128da `1.069x` bo‘ldi va max logit farqi `0.007057` chiqdi;
bu ham defaultga olinmadi. Batafsil:
`results/RUNTIME_NATIVE_SMALL_BATCH_BASELINE_20260908.md`.

Native branch recheckda CUDA Graph logit error `0.0` bo‘ldi, lekin batch-1
`1.009x` va batch-128 `0.986x` bo‘lib, faqat taxminan `1.4%` batch-128
mikro-foyda berdi. `torch.compile` qayta sinovida Triton topilmadi; bu Windows
toolchain cheklovi, model rejection emas. Fused decode kernel muammosi ochiq
qoladi.

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
100M va 300M control bilan tekshirish; route-causal diagnostic o‘tkazildi va
prefix-preserving transition window (`H1`) 100M ikki-seed pilotda `−0.44/−0.13
pp` bo‘lib, `NOT PROMOTED` qilindi. 500M faqat scale-control sifatida qoladi.
Keyingi sinov circuit output → recurrent state/final-target causal interface’iga
qaratiladi. Batafsil:
`results/P003_ROUTE_CAUSAL_DIAGNOSTIC_20260907.md`.

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

**Frontier recheck (2026-09-10):** Oltita frozen checkpoint bir xil route-audit
protokolida qayta tekshirildi. 100M seed17/18da bankning `67.3%/64.8%` qismi
ishlatilgan va dead fraction `32.71%/35.20%` bo‘lgan. 300M’da usage
`25.5%/24.9%`, dead `74.52%/75.07%`; 500M’da usage `24.2%/23.8%`, dead
`75.82%/76.17%` bo‘ldi. Between-task union Jaccard 100M’dagi
`0.0488/0.0513`dan 500M’da `0.0126/0.0168`gacha tushdi. Ikki seedda ham bir
xil yo‘nalish bor: capacity-only scale yangi foydali computationga aylanmay,
route coverage va reuse parchalanmoqda. Bu bankdagi barcha circuitlar
foydasizligini isbotlamaydi, ammo 700M/1B’ga o‘tishdan avval coverage/reuse
muammosini hal qilish kerakligini kuchaytiradi.

**Holat:** `DIAGNOSTIC CONFIRMED; CAPACITY-ONLY SCALE DEFERRED`.
**Batafsil:** `results/P003_ROUTE_FRONTIER_RECHECK_20260910.md`.

**Qo‘shimcha nazorat:** frozen prefix clamp 100M/300M/500M seed17/18da izchil
quality foydasi bermadi; katta bankning o‘zi yetarli sabab emas. Training-time
5% tree exploration esa 300M continuationda dead/CE ayrim hollarda yaxshilangan
bo‘lsa-da, hard accuracy ikki seed o‘rtachasida `−0.13 pp` bo‘ldi. Demak
coverage’ni mexanik oshirish ham foydali computationga aylanmadi.

**Holat:** prefix clamp `REJECTED AS QUALITY FIX`; route exploration
`REJECTED FOR ADOPTION`. Keyingi sinov route exposure’ni yana oshirish emas,
circuit correctionning recurrent state/outputga ta’sirini operation composition
bilan bog‘laydigan minimal interface bo‘lishi kerak. 500M scale-control sifatida
saqlanadi, defaultga ko‘chirilmaydi.

**Batafsil:** `results/P003_ROUTE_CAPACITY_CLAMP_AUDIT_20260910.md` va
`results/P003_ROUTE_EXPLORATION_CONTINUATION_AUDIT_20260910.md`.

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

Post-GRU correction residual (`post_correction_residual_scale=β`) ham
inference-only tekshirildi. `β=1.0` seed17’da `+1.25 pp`, seed18’da
`−0.42 pp` natural accuracy berdi; CE va route-replay sensitivity ham
qarama-qarshi bo‘ldi. Shuning uchun correction’ni GRU’dan keyin bevosita
qo‘shish mavjud checkpoint uchun barqaror quality fix emas va training/defaultga
qabul qilinmadi. Opt-in API saqlandi.

**Residual audit:** `results/P007_POST_CORRECTION_RESIDUAL_AUDIT.md`.

Route-final-target auxiliary loss ham tekshirildi. Har bir bajarilgan
circuit delta `model.output(delta)` orqali final targetga auxiliary CE bilan
bog‘landi; inference, active budget va circuit bank o‘zgarmadi. 20M
coverage-matched continuationda `w=0.10` ikki seedda `+0.495/+0.260 pp`
(mean `+0.378 pp`) berdi, ammo adoption gate `+2 pp`ga yetmadi. `w=0.05`
mean `+0.260 pp` bo‘lib seedlar orasida qarama-qarshi, `w=0.25` esa
`−0.195 pp` bo‘ldi. Demak bu route/state interface uchun zaif signal, lekin
muammoni yechgan ishonchli arxitektura emas; defaultga kiritilmadi va 100M ga
scale qilinmadi.

**Route-final-target audit:** `results/P007_ROUTE_FINAL_TARGET_AUDIT.md`.

Training bilan `post_correction_residual_scale=1.0` ham ikki seedda tekshirildi:
seed17 accuracy `−0.104 pp`, seed18 `+0.078 pp`, mean `−0.013 pp`; CE mean
`+0.003219` yomonlashdi. Demak inference-only bypassdagi qarama-qarshi signal
training bilan ham tuzalmadi. Route-state bypass oilasi `REJECTED FOR ADOPTION`,
100M/300Mga scale qilinmaydi. Batafsil:
`results/P007_POST_CORRECTION_RESIDUAL_TRAINING_AUDIT_20260908.md`.

**One-step route causality audit (2026-09-08):** 100M/300M seed17/18da faqat
bitta internal step route’i almashtirilganda route-delta L2 `1.53–1.90`, keyingi
query L2 `0.56–0.96`, final logit L2 `0.59–1.03` bo‘ldi. Demak route correction
recurrent state’ga causal ravishda yetib boradi; u butunlay yutilib ketmayapti.
Shu bilan birga final CE `−0.00431…+0.00211`, hard accuracy esa
`−1.25…+0.42 pp` diapazonda seed/taskga qarab o‘zgardi. P-007 ochiq qoladi,
lekin bottleneck route-state interface’dan ko‘ra candidate retrieval, subset
regret, specialization va final classifier margin tomoniga siljidi. Bypass va
correction-scale variantlarini yana scale qilmaymiz. Batafsil:
`results/P007_ROUTE_STEP_CAUSALITY_AUDIT_20260908.md`.

### C-P002-EXPOSURE-WARMUP-001 — Initial task-stable route exposure

**Status:** `REJECTED`
**Muammo:** P-002 / P-003
**Natija:** seed17/18 held-out mean accuracy delta `+0.768 pp`; route NMI delta `+0.01827`; route specialization delta `+0.01578`; counterfactual NMI/specialization deltas `-0.02349`/`-0.01738`. Birinchi `1000` qadamda task-stable route, undan keyin oddiy learned hard router ishladi; router va circuit body o‘zgarmadi.

### C-P003-CLONE-INIT-001 — Parent-cloned initialization of new bank rows

**Status:** `REJECTED`
**Muammo:** P-002 / P-003
**Natija:** seed17/18 clone minus random held-out mean accuracy `+0.508 pp`; route NMI delta `-0.00525`; route specialization delta `-0.00821`; dead fraction delta `-0.00285`. Yangi 100M bank qatorlari top `64` parent circuitdan noise bilan initsializatsiya qilindi; qolgan protocol random arm bilan bir xil.

### C-P002-SHARED-RESIDUAL-001 — Shared reusable residual primitive

**Status:** `REJECTED`
**Muammo:** P-002 / P-003 / P-007
**Natija:** seed17/18 shared-residual held-out mean accuracy delta `-0.299 pp`; counterfactual NMI/specialization/positive-advantage deltas `-0.02001`/`-0.00632`/`-0.02865`; dead fraction delta `+0.00000`. Shared rank-8 primitive V0’ga opt-in sifatida qo‘shildi, mustaqil per-circuit residual va router saqlandi.

**Fourier/algebraic register bridge sinovi (2026-09-08):** intermediate
`step_logits`ni learned 64-vector embedding o‘rniga input numeric encoder’da
ishlatilgan fixed mod-64 Fourier koordinatalariga o‘tkazib, `13 → state_dim`
projection orqali keyingi query’ga qaytarish tekshirildi. Bridge-only ikki
seedda final accuracy `−0.469/−0.313 pp`, o‘rtacha `−0.391 pp`, mean CE
`+0.001305` bo‘ldi. Bridge+stage `−0.313/+0.365 pp`, o‘rtacha `+0.026 pp`,
mean CE `−0.000172` berdi. Qo‘shimcha parametr faqat `5,376`; shunga qaramay
`chain3`, `compose_add_mul`, `state_machine`da barqaror composition foydasi
chiqmadi. Fourier representation reuse **REJECTED FOR ADOPTION**; P-004
`ACTIVE` qoladi va keyingi yo‘l representationni ko‘chirish emas, haqiqiy
operator/state dataflow yoki circuit specializationni nishonga olishi kerak.

**Batafsil:** `results/P004_FOURIER_REGISTER_BRIDGE_AUDIT_20260908.md`.

**Recurrent state information probe (2026-09-08):** 100M/300M/500M seed17/18
frozen checkpointlarda query state oldingi intermediate targetni chance
`1.5625%` emas, linear probe’da step1 `59.59%`, step2 `45.14%` va nonlinear
probe’da mos ravishda `67.27%`, `51.32%` held-out accuracy bilan olib yurishi
ko‘rsatildi. Direct stage output `67.60%/60.04%` bo‘lib, nonlinear probe’dan
ustun qoldi. Demak ma’lumot state’da butunlay yo‘qolmayapti, ammo capacity
oshganda signal monotonik kuchaymayapti; P-004ning asosiy ceilingi state’dan
keyingi operation uchun foydali composition olishda.

**Batafsil:** `results/P004_STATE_INFORMATION_PROBE_AUDIT_20260908.md`.

**State-history skip sinovi (2026-09-08):** oldingi recurrent state’larni
step-2 query’ga parameter-free normalized sum sifatida berish 20M seed17/18/19
5k continuationda history+stage treatment sifatida tekshirildi. Final accuracy
delta `+0.260/+1.146/+0.938 pp`, o‘rtacha `+0.781 pp`; matching history’siz
stage-only control o‘rtachasi `+0.139 pp` bo‘ldi. History interaction accuracy
taxminan `+0.642 pp`, lekin CE interaction `+0.007819` yomonlashdi; history-only
2k o‘rtachasi `−0.078 pp`. Shuning uchun bu faqat zaif diagnostik signal:
opt-in qoldi, default va scale o‘zgarmadi. P-004 `ACTIVE` qoladi.

**Batafsil:** `results/P004_STATE_HISTORY_AUDIT_20260908.md`.

**Task-scaled history sinovi (2026-09-08):** har bir 15 task uchun alohida
o‘rganiladigan history read-scale qo‘shildi. 20M seed17/18 2k matching
continuationda final accuracy deltalari `+0.885/-0.729 pp`, o‘rtachasi
`+0.078 pp`; mean CE `+0.00002679` yomonlashdi. Stage-0 ikkala seedda oshgan,
ammo keyingi stage’lar sign-flip qilgan. Shuning uchun task-specific scale
`REJECTED FOR ADOPTION`; 100M+ ga scale qilinmaydi. P-004 `ACTIVE` qoladi va
keyingi yo‘l operation-specific state write/read yoki aniq intermediate-value
dataflow bo‘ladi.

**Batafsil:** `results/P004_STATE_HISTORY_TASK_SCALE_AUDIT_20260908.md`.

**DynamicRegister state-trace audit (2026-09-08):** 0--7 non-modular flat
checkpointlarda direct intermediate accuracy stage-0/1da `100%/100%`,
stage-2/3da `83.89%/66.70%` bo‘ldi. Post-accumulator state bilan haqiqiy
intermediate value korrelyatsiyasi seed17/18da mos ravishda
`1.000/1.000 → 0.905/0.888 → 0.503/0.533 → 0.207/-0.012` ga kamaydi.
Bu output headning yagona muammo emasligini va depth-4da takroriy state
composition buzilayotganini ko‘rsatadi; P-004 `ACTIVE` qoladi.

**Keyingi opt-in test:** learned operation-conditioned bilinear scalar lane
(`old`, `operand`, `old*operand`, `bias`)ni factorized-output controlga qo‘shish.
Bu arithmetic oracle emas; koeffitsientlar o‘rganiladi. Bank yoki router
sig‘imi oshirilmaydi. **Batafsil:**
`results/V0_200_DYNAMIC_NONMOD_STATE_TRACE_AUDIT.md`.

**Persistent scalar read sinovi (2026-09-09):** learned scalar lane’ni query
va outputdan tashqari keyingi operation `read_accumulator`iga ham qo‘shish
tekshirildi. Factorized control ikki seedda `71.97%`, scalar-only treatment
`72.36%` (`+0.39 pp`, seed-unstable), persistent-read treatment esa `71.68%`
(`−0.29 pp`) berdi. Scalar injection pointlarini ko‘paytirish
**REJECTED FOR ADOPTION**; P-004 `ACTIVE` qoladi. Keyingi ish alohida
persistent algebraic value packet/authoritative transition bo‘lishi kerak.

**Batafsil:** `results/V0_201_DYNAMIC_NONMOD_SCALAR_READ_AUDIT.md`.

**Authoritative scalar packet sinovi (2026-09-09):** scalar register’ni dense
state o‘rniga authoritative qilish seed17da 1,000 qadamda seen accuracy
`27.73%`, held-out `5.08%`, loss `3.59` berdi; 1,500 qadamda loss `3.61`ga
plateau qildi. **REJECTED AS A LEARNING CONFIGURATION.** Bu persistent
packetning o‘zini emas, numeric contract loss’siz authority berishni rad
qiladi. P-004 `ACTIVE`; keyingi sinov bo‘lsa packetga normalized direct value
loss beriladi, capacity oshirilmaydi.

**Batafsil:** `results/V0_202_DYNAMIC_NONMOD_AUTHORITATIVE_VALUE_AUDIT.md`.

**Authoritative value contract sinovi (2026-09-09):** scalar packetga
normalized direct stage loss qo‘shildi. Weight `1`da seed17 1,000 qadamda
seen/held-out `27.34%/5.08%`, weight `100`da `26.56%/6.25%` va loss `4.42`
bo‘ldi. **REJECTED.** Bu scalar packet family’sini yopadi: keyingi yo‘l
dense writerdagi explicit residual state update; capacity oshirilmaydi.

**Batafsil:** `results/V0_203_DYNAMIC_NONMOD_VALUE_CONTRACT_AUDIT.md`.

**Dense state residual sinovi (2026-09-09):** `new = old + 0.25 * proposal`
update seed17da 1,000 qadamda `65.63%`, 3,000 qadamda `66.99%` held-out berdi;
matched factorized control `71.09%`. Depth-4 `53.52%` bo‘lib, controldagi
`59.38%`dan `5.86 pp` past. Seen depths `100%` bo‘lsa ham depth-transfer
regressiyasi sabab **REJECTED FOR ADOPTION**. Scalar va oddiy residual state
preservation oilasi yopildi; P-004 `ACTIVE`, keyingi yo‘l genuinely structured
transition yoki FFN/circuit-transplant lane.

**Batafsil:** `results/V0_204_DYNAMIC_NONMOD_STATE_RESIDUAL_AUDIT.md`.

**Authoritative packet 9k stress (2026-09-10):** V0.202 authoritative scalar
packetga to‘liq 9,000 qadam berilganda ham seed17/18 held-out o‘rtachasi
`65.72%` bo‘ldi; depth-3 `70.70%`, depth-4 `60.74%`. Seen-depth o‘rtachasi
`89.79%` bo‘lsa-da, factor rows deyarli to‘liq ishlatilgan vaqtda ham
compositional transfer qaytmadi. Bu 1,000-qadamdagi `5.08%` natijadan ancha
yaxshi, lekin non-authoritative factorized reference’dagi `71.97%`dan past.
Shuning uchun scalar packet authority’ni ko‘proq qadam yoki scale bilan
davom ettirish **REJECTED FOR ADOPTION AND SCALING**. P-004 `ACTIVE`; keyingi
yo‘l yana scalar injection/contract/residual emas, genuinely structured
value/state representation yoki explicit algebraic transition bo‘lishi kerak.

**Batafsil:** `results/V0_205_DYNAMIC_NONMOD_AUTHORITATIVE_VALUE_9000_AUDIT.md`.

**Algebraic state primitive (2026-09-10):** ikki koordinatali fixed `x,x^2`
packetni learned dense query/output bilan birga olib yurish seed17/18da
held-out o‘rtachani `71.97%`dan `78.52%`ga ko‘tardi (`+6.55 pp`); depth-4
o‘rtacha foyda `+10.74 pp`, total/active budget esa `7.35M/2.05M` atrofida
qoldi. Bu hozirgi P-004 uchun kuchli ijobiy signal, ammo semantic transition
fixed va normalization bounded bo‘lgani uchun hali default qilinmaydi. Keyingi
gate operand diapazonini `0--15`ga kengaytirish va circuit yo‘lining haqiqiy
zarurligini alohida o‘lchashdir.

**Batafsil:** `results/V0_206_DYNAMIC_NONMOD_ALGEBRAIC_STATE_AUDIT.md`.

**Algebraic state range stress (2026-09-11):** polynomial2 `x,x^2` packet
operandlar `0--15`ga kengaytirilganda seed17/18 held-out o‘rtachasi
`64.45%`, depth-4 `53.22%` bo‘ldi; `0--7`da esa `78.52%` va `70.70%` edi.
Seen-depth fit `99.95%`, factor-row usage keng, shuning uchun bu router
starvation emas, range/readout transfer muammosi. Polynomial2 packet hozircha
bounded diagnostic sifatida qoladi; universal quality yoki scale yechimi deb
qabul qilinmaydi. Keyingi yo‘l range-aware value codec/readout, bankni
ko‘paytirish emas.

**Batafsil:** `results/V0_207_DYNAMIC_NONMOD_ALGEBRAIC_STATE_RANGE_STRESS.md`.

**Range-aware algebraic/Fourier bridge (2026-09-11):** polynomial2 packetga
`128`, `16,384`, `524,288` periodli 42 ta Fourier feature qo‘shilganda
operand `0--15` held-out o‘rtachasi `64.45%`dan `94.29%`ga, depth-4 esa
`53.22%`dan `92.09%`ga ko‘tarildi; qo‘shimcha learned projection faqat
`16,128` parametr. Bu P-004 uchun kuchli ijobiy signal va muammo value-to-
readout codingda ekanini ko‘rsatadi, ammo periodlar base-128 codecga
moslanganligi sabab hali universal yechim emas. Keyingi gate `0--31`, scale
oshirish emas.

**Batafsil:** `results/V0_208_DYNAMIC_NONMOD_ALGEBRAIC_FOURIER_AUDIT.md`.

**0--31 output-base nazorati (2026-09-11):** range-aware Fourier bridge bilan
`base=512` ikki seedda held-out o‘rtacha `78.81%`, depth-4 `73.83%` berdi;
base-1024 reference `75.10%`/`69.92%`dan `+3.71/+3.91 pp` yuqori. Biroq
active parametr `14.93M`dan `27.35M`ga, training vaqti taxminan `19%`ga oshdi.
Base2048 `69.82%`/`62.89%`, base4096 `62.30%`/`53.32%` bo‘ldi. Base512
**RETAINED AS OPT-IN QUALITY REFERENCE**, default qilinmaydi; P-004 active
qoladi va keyingi ish base512 sifatini pastroq active budgetga olib tushuvchi
shared/low-rank value codec bo‘ladi.

**Batafsil:** `results/V0_209_DYNAMIC_NONMOD_ALGEBRAIC_FOURIER_0_31_OUTPUT_BASE_AUDIT.md`.

**Shared low-rank value codec (2026-09-11):** base512 output head oldidan
shared `384→128` projection qo‘shilganda held-out o‘rtacha `85.06%`, depth-4
`81.05%` bo‘ldi. Bu base-1024 reference’dan `+9.96/+11.14 pp`, full base512
variantdan `+6.25/+7.23 pp` yuqori; total/active parametr `15.79M/10.49M`ga
tushdi. **LEADING OPT-IN SIGNAL; DEFAULT EMAS.** P-004ning asosiy natijasi
endi faqat router emas, value-state→readout codec ekanini ko‘rsatmoqda. Ochiq
ishlar: rank `64/128/256` ablation, shifted offset va kengroq operand stress;
700M/1B scale hozircha qilinmaydi.

**Batafsil:** `results/V0_210_DYNAMIC_NONMOD_ALGEBRAIC_FOURIER_SHARED_CODEC_AUDIT.md`.

**Target-offset robustness (2026-09-11):** shared rank-128 base512 codec
offset `1,048,576`dan `2,097,152`ga ko‘chirilganda ikki seedli held-out
o‘rtacha `85.06%`dan `84.86%`ga, depth-4 esa `81.05%`dan `82.42%`ga o‘zgardi;
parametrlar `15.79M/10.49M` bo‘lib qoldi. **ROBUSTNESS GATE PASSED.** Bu
codecning bitta absolut target offsetni yodlab qolmaganini ko‘rsatadi. P-004
active qoladi; keyingi muhim screen operandlarni `0--63`ga kengaytirish,
700M/1B scale emas.

**Batafsil:** `results/V0_211_DYNAMIC_NONMOD_ALGEBRAIC_FOURIER_SHARED_CODEC_OFFSET_ROBUSTNESS.md`.

**Operand range `0--63` depth-3 gate (2026-09-11):** rank128 shared codec
ikki seedda held-out o‘rtacha `85.74%`, active `10.49M` berdi; matched 0–31
depth-3 reference’dan farq `−3.32 pp`, ya’ni signal qulamadi. 0–63 depth-4da
oldingi `67M` class config yetarli emasligi guard bilan aniqlandi: targetlar
`225.6M`gacha chiqdi. Bu quality failure emas, ikki-digit output codec
chegarasi. P-004 active; keyingi ish 1B-class targetni uchta kichik digit head
va shared low-rank projection bilan ifodalash.

**Batafsil:** `results/V0_212_DYNAMIC_NONMOD_ALGEBRAIC_FOURIER_RANGE63_DEPTH3_AUDIT.md`.

**Three-digit range-safe codec (2026-09-11):** `2^30` class space uchun
base1024 uchta digit head qo‘shildi. Rank128 ikki seedda held-out `63.28%`,
depth-4 `55.08%`; rank256 `65.63%`/`58.59%` bo‘ldi. Parametr juda kam
(`2.37M/2.81M active`), lekin quality leading two-digit rank128 yo‘lidan ancha
past. **REJECTED FOR QUALITY ADOPTION.** Muammo faqat class range emas:
independent digit logits unseen-depth compositionni yo‘qotmoqda. Keyingi
codec cross-digit interactionga ega bo‘lishi kerak; bank/700M/1B scale emas.

**Batafsil:** `results/V0_213_DYNAMIC_NONMOD_THREE_DIGIT_CODEC_AUDIT.md`.

**Three-digit Fourier-base alignment (2026-09-11):** output base1024ga
Fourier-base1024 moslanganda ikki seedli held-out `63.28%`dan `67.97%`ga,
depth-4 `55.08%`dan `58.40%`ga ko‘tarildi; active budget `2.37M` bo‘lib qoldi.
Bu secondary improvement, lekin leading two-digit rank128 sifatiga yaqin emas.
**RETAINED AS DIAGNOSTIC; QUALITY ADOPTION REJECTED.** Asosiy ochiq muammo
independent digit headlarda cross-digit/carry interaction yo‘qligi; model
scale oshirish emas, shu interactionni qurish kerak.

**Batafsil:** `results/V0_214_DYNAMIC_NONMOD_THREE_DIGIT_FOURIER_BASE_AUDIT.md`.
