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

2026-09-10 distribution-shift auditida 10k qadamli ordered shared-route-key
checkpointlar qayta tekshirildi. Ikki seed o‘rtachasida 500M 300Mdan oddiy
balanced streamda `+0.417 pp`, combination hold-outda `+0.434 pp` yaxshi chiqdi.
Lekin low-edge `[0,7]` probe’da `−0.204 pp`, high-edge `[56,63]` probe’da
`−0.490 pp` bo‘ldi. Edge probe’larda ikkala modelda ham dead traffic
`~46–55%`gacha oshdi. Demak 500M sig‘imi foydali signal bera oladi, ammo yangi
capacity distribution shift ostida barqaror specializationga aylanmayapti;
P-003 yechilgan emas.

**Batafsil:** `results/P003_NATIVE_OOD_DISTRIBUTION_AUDIT_20260910.md`.

Keyingi ikki-seedli 10k auditida step-adapter bilan birga training oqimining
25%ini low-edge `[0,7]` va high-edge `[56,63]` rejimlariga teng ajratish sinab
ko‘rildi. Oddiy evaluatorda uniform accuracy `77.266% → 77.865%` va hard-task
mean `46.658% → 48.926%` oshdi; low/high edge probe’larda mos ravishda
`+17.340/+22.097 pp` chiqdi. Demak P-003dagi edge regressiyaning katta qismi
rare joint-value coverage bilan bog‘liq bo‘lishi mumkin. Bu training recipe,
arxitektura yechimi emas; 500Mda qayta tasdiqlanmaguncha default o‘zgarmaydi.

**Batafsil:** `results/P003_NATIVE_EDGE_MIX_AUDIT_20260910.md`.

### C-P003-NATIVE-WARMSTART-002 — Non-stable factor-grid warm-start

**Status:** `REJECTED AS SUFFICIENT CAPACITY FIX; RETAINED AS CONTROL`
**Muammo:** P-003 / P-007

300M 10k checkpointidan 500M factor gridiga reusable factor rows va pair-mix
ko‘chirilib, 3k davom ettirildi. 24-batch evaluator’da 500M warm-startning
uniform accuracy mean qiymati `80.707%`, hard-task mean `54.546%` bo‘ldi;
300Mning aynan 3k continuation nazorati `80.920%` va `54.796%` berdi.
Demak warm-start route dead fractionni `24.17% → 13.54%` kamaytirdi, ammo
quality bo‘yicha yetarli ustunlik bermadi. Muammo factor rowsning o‘zida emas,
virtual address va route-tree semantics saqlanmaganida bo‘lishi mumkin.

**Batafsil:** `results/diagnostic_native_warmstart_vs_continued_20260910.json`.

### C-P003-NATIVE-STABLE-PREFIX-001 — Stable virtual-address staged growth

**Status:** `PROMISING OPT-IN — QUALITY-NEUTRAL UNDER MATCHED BUDGET; LONG RUN OPEN`
**Muammo:** P-003 / P-007

Stable-prefix address map 300Mning birinchi 22,800 virtual pairlarini 500Mda
aynan saqladi; yangi 15,800 address va oltinchi router leveli 3k warm-updan
keyin ochildi. Ikki seedli, bir xil 24-batch evaluator’da stage-2 500M
300M 3k continuationga nisbatan uniform accuracy’ni `80.920% → 82.040%`
(`+1.120 pp`), hard-task mean’ni `54.796% → 56.695%` (`+1.899 pp`),
combination holdoutni `81.419% → 82.357%` (`+0.938 pp`) yaxshiladi. Low/high
edge ham mos ravishda `+0.538/+0.373 pp` bo‘ldi. Dead circuit fraction
`24.17% → 14.21%` tushdi.

Initial comparisondagi ijobiy farq stage-2 variantda jami 6k qo‘shimcha step,
300M controlda esa 3k qo‘shimcha step bo‘lgani uchun compute-confounded edi.
Clean 6k-versus-6k evaluatorda stable-prefix 500M uniformni `82.080% →
82.040%` (`−0.040 pp`), hard-task meanni `56.767% → 56.695%` (`−0.072 pp`)
berdi; combination holdout `+0.082 pp`, combination hard-task `+0.206 pp`
bo‘ldi. Shu bilan birga dead fraction `26.22% → 14.21%` tushdi va 197 factor
row ishlatildi. Demak bu hozircha katta quality jump emas, balki qo‘shimcha
capacityni sifatni buzmasdan route’ga kiritish signali. Long stage-2 va
compute-matched 7k continuationda ham 500M uniform `84.485%`, 300M esa
`84.546%` berdi (`−0.061 pp`); hard-task mean `62.080%` va `62.388%` bo‘ldi
(`−0.308 pp`). Dead fraction esa `26.80% → 15.62%` tushdi. Bu extra rows
route’ga kirayotganini, lekin quality bottleneck representation yoki
difficulty-conditioned active compute tomonida ekanini ko‘rsatadi. 700M/1Bga
o‘tish muzlatilgan, variant default emas, opt-in sifatida saqlandi.

**Batafsil:** `results/P003_NATIVE_STABLE_PREFIX_GROWTH_AUDIT_20260910.md`.

### C-P003-NATIVE-ACTIVE-WIDTH-001 — Fixed K=8 may cap hard-task quality

**Status:** `PROMISING DIAGNOSTIC — DYNAMIC WIDTH OPEN`  
**Muammo:** P-003 / P-007

Stable-prefix 500M checkpointida `active_circuits=8`ni `16`ga oshirish ikki
seedli, 24-batch OOD screen’da uniform exact accuracy’ni `82.040% → 82.999%`
(`+0.959 pp`), uniform hard-task mean’ni `56.695% → 58.583%` (`+1.888 pp`),
combination hard-task mean’ni `57.107% → 58.952%` (`+1.845 pp`) oshirdi.
Dead virtual-circuit fraction `14.21% → 1.52%` tushdi. Bu fixed K=8 joriy
representation uchun qiyin tasklarda active-compute ceiling bo‘lishi mumkinligi
haqida hozirgi eng kuchli ijobiy signal.

Ammo K=16 barcha sample’larda majburiy ishladi: active circuit parameters
taxminan `202,768 → 405,536`, active-parameter estimate `~1.70M → ~2.31M`,
throughput `~1,838 → ~1,160 samples/s` bo‘ldi. Shuning uchun bu selective
solution emas va compute-matched scaling dalili emas. K=16 default qilinmadi;
keyingi muammo — hard tasklarda 8, kerak bo‘lganda 16 slotni tanlaydigan,
haqiqiy sparse dispatch bilan ishlaydigan difficulty-conditioned dynamic-width
router. Easy task K=8da qolishi, hard task esa foydasi bo‘lsa K=16ga chiqishi
kerak; logical mask va real runtime cost alohida o‘lchanadi.

**Batafsil:** `results/P003_NATIVE_ACTIVE_WIDTH_AUDIT_20260910.md` va
`results/diagnostic_native_active_width_20260910.json`.

### C-P003-NATIVE-DYNAMIC-WIDTH-001 — Fixed entropy proxy saves too little compute

**Status:** `PROMISING PROTOTYPE — LEARNED WIDTH OPEN`  
**Muammo:** P-003 / P-007

K=16 trained checkpointida top-8 route weight entropy bilan sample/step
darajasida K=8 yoki K=16 dispatch qilindi. Threshold `0.995`da uniform mean
active width `16 → 14.64` (`~8.5%` slot reduction), wide dispatch fraction
`83.0%` bo‘ldi. Shu bilan birga uniform exact accuracy `82.999% → 82.990%`
(`−0.009 pp`), uniform hard-task mean `58.583% → 58.573%` (`−0.010 pp`),
combination hard-task mean `58.952% → 58.930%` (`−0.022 pp`) bo‘ldi. Demak
actual executed-slot accounting bilan sifat deyarli saqlandi, ammo entropy proxy
katta tejash bermadi; low-edge’da ham keng yo‘l ko‘p tanlandi.

`selected_ids` va `executed_selected_ids` alohida saqlandi, shuning uchun
router qarori va real circuit computation aralashtirilmaydi. Bu variant default
emas. Keyingi ish — K=8 va K=16 orasidagi o‘lchangan loss gapidan cost-aware
learned width head/predictor o‘qitish va held-out hard-task gate bilan tekshirish.

**Batafsil:** `results/P003_NATIVE_DYNAMIC_WIDTH_AUDIT_20260910.md` va
`results/diagnostic_native_dynamic_width_threshold0995_20260910.json`.

### C-P003-NATIVE-DYNAMIC-WIDTH-002 — Oracle shows real selective-width headroom

**Status:** `POSITIVE ORACLE HEADROOM — LEARNED WIDTH PREDICTOR JUSTIFIED`
**Muammo:** P-003 / P-007

Bir xil 500M K=16 checkpoint uchun fixed K=8 va fixed K=16 final logitslari
har bir example’da alohida hisoblanib, `lambda * width_fraction` penalti bilan
oracle tanlandi. Ikki seedli 24-batch OOD screen’da `lambda=0.05`da uniform
active width `51.3%`, exact accuracy `83.060%`, hard-task mean `58.681%` bo‘ldi;
fixed K=16 control `82.999% / 58.583%` edi. Combination holdoutda active
width `51.2%`, exact `83.168%`, hard mean `59.071%` bo‘ldi. `lambda=0.10`da
width taxminan `50.5%`ga tushib, sifat hamon controlga yaqin qoldi.

Bu entropy proxy natijasidan ancha yaxshi: u faqat `~8.5%` width tejagan edi.
Ammo oracle ikkala widthni ham oldindan hisoblaydi va deployable emas; recurrent
state K=8/K=16 tanlovidan keyin farq qiladi. Demak bu sifat/compute ceiling,
router tayyor degani emas. Keyingi ish — paired K=8/K=16 loss-gapdan kichik
cost-aware width head o‘qitish, keyin disjoint seedda bitta yo‘l dispatchini
hard-task regret, mean executed width va real runtime bilan tekshirish.

**Batafsil:** `results/P003_NATIVE_DYNAMIC_WIDTH_ORACLE_AUDIT_20260910.md` va
`results/diagnostic_native_dynamic_width_oracle_20260910.json`.

### C-P003-NATIVE-DYNAMIC-WIDTH-003 — Learned width predictor preserves quality

**Status:** `PROMISING OPT-IN — QUALITY GATE PASSED; LONGER RUN OPEN`
**Muammo:** P-003 / P-007

Frozen 500M K=16 modelidan paired K=8/K=16 step-loss label bilan `384→1`
width head o‘qitildi. Ikki seedli, 24-batch OOD screen’da learned dispatch
uniform exact accuracy’ni K=16 controlga nisbatan `82.999% → 83.008%`
(`+0.009 pp`), hard-task mean’ni `58.583% → 58.594%` (`+0.011 pp`) berdi.
Combination holdout delta `+0.004 pp`, high-edge delta `+0.013 pp`, low-edge
delta `0.000 pp` bo‘ldi.

Actual executed-slot accounting bo‘yicha uniform mean active width `8.99/16`
(`56.2%`), wide K=16 fraction `12.4%`; combinationda `8.98/16`, edge
probelarda `8.4–8.5/16`. Bu entropy proxydagi `~8.5%` tejashdan ancha yaxshi
va oraclening taxminan yarim-width nuqtasiga yaqin. Head faqat `385` yangi
parametr qo‘shadi; circuit bank/body o‘zgarmadi.

Seed19 scratchdan mustaqil tekshirildi: uniform `66.476% → 66.484%`,
combination `66.623% → 66.597%`, low-edge `91.059% → 90.955%`, high-edge
`88.568% → 88.828%`; learned active width `8.2–8.8/16` oralig‘ida qoldi.
U warm-start seedlar bilan capacity comparison’ga qo‘shilmaydi, lekin predictor
signalining faqat seed17/18ga xos emasligini ko‘rsatadi. Uzoqroq continuation,
K=8ga noto‘g‘ri yuborish regreti va body bilan joint training hali ochiq.
Learned-width checkpointlari opt-in saqlandi.

**Batafsil:** `results/P003_NATIVE_LEARNED_WIDTH_AUDIT_20260910.md` va
`results/diagnostic_native_learned_width_20260910.json`.

### C-P003-NATIVE-DYNAMIC-WIDTH-004 — Grouped dispatch overhead reduces width savings

**Status:** `POSITIVE RUNTIME SIGNAL — OPT-IN ONLY; PREFIX-SPLIT REJECTED; KERNEL FUSION OPEN`
**Muammo:** P-003 / Runtime

RTX 3060 timing screen’da balanced batch 480, 5 warm-up va 20 synchronized
CUDA repeat bilan uch seed o‘rtachasida fixed K=8 `32.39 ms`, fixed K=16
`55.44 ms`, learned K=8/16 `39.89 ms` chiqdi. Learned path fixed K=16ga
nisbatan `~28.1%` latency kamayishi va `~39.0%` throughput oshishini berdi;
mean active width `8.58/16` (`~46.4%` slot reduction). Demak circuit
computation kamayishi real tezlikka aylanayapti, lekin narrow/wide samplelarni
alohida dispatch qilish overheadi ideal width tejamining bir qismini yutmoqda;
learned path fixed K=8dan `~23.1%` sekinroq.

Batch=15da `dynamic_width_min_batch=32` guard K=16ga fallback qildi; learned
`7.51 ms`, fixed K=16 `7.33 ms` (`~2.3%` overhead) bo‘ldi. Unguarded oldingi
variant `~11.8 ms` bo‘lgan, shuning uchun guard small-batch regressiyasini
sezilarli kamaytirdi.

960-example balanced batchdagi alohida uch-seed timingda fixed K=8 `65.69 ms`,
fixed K=16 `105.77 ms`, learned `67.84 ms` bo‘ldi: K=16ga nisbatan `35.9%`
latency kamayishi, K=8ga nisbatan faqat `3.3%` overhead. Mean active width
`8.67/16`. Demak foyda bitta batch o‘lchamiga xos emas, lekin kernel launch
overheadi hali ham mavjud.

Bu hali default emas. Uzunroq continuation va kernel fusion profili kerak.
Keyingi optimizatsiya — grouped-dispatch kernel fusion; sifat va route qarori
o‘zgarmaydi.

**Batafsil:** `results/P003_NATIVE_DYNAMIC_WIDTH_RUNTIME_AUDIT_20260910.md`,
`results/diagnostic_native_width_runtime_all3_20260910.json` va
`results/diagnostic_native_width_runtime_batch960_20260910.json`.

96-batch-per-condition matched OOD controlda learned-minus-fixed-K=16 exact
deltalar uniform `+0.012 pp`, combination `−0.019 pp`, low-edge `−0.038 pp`,
high-edge `+0.067 pp`; hard-task deltalari `−0.096…+0.168 pp` oralig‘ida qoldi.
Bu uzoqroq tekshiruv ham sifatda sistematik pasayish ko‘rsatmadi.

**Sifat nazorati:** `results/diagnostic_native_learned_width_long96_20260910.json`
va `results/diagnostic_native_fixed16_long96_20260910.json`.

Prefix-split variant additive bank uchun algebraik jihatdan to‘g‘ri va grouped
variant bilan output/route ID testlari mos chiqdi, lekin uch seedli 480-example
timingda grouped `39.61 ms`, prefix-split `41.64 ms` (`+5.1%`) bo‘ldi. Shuning
uchun bu fusion yo‘li hozirgi backend uchun **REJECTED AS RUNTIME IMPROVEMENT**;
opt-in kod faqat qayta tekshirish uchun qoldirilgan.

**Prefix-split dalili:** `results/diagnostic_native_width_prefix_split_480_20260910.json`.

### C-RUNTIME-NATIVE-COMPILE-001 — Local Inductor/Triton compiler blocker

**Status:** `BLOCKED LOCALLY — STATIC/FUSED PATH REQUIRED`
**Muammo:** Runtime track / P-003

Native fixed K=8, fixed K=16 va learned grouped K=8/K=16 uchun PyTorch
`2.6.0+cu124` `torch.compile(mode="reduce-overhead")` probe qilindi. Uchala
variantda ham eager timing ishladi, lekin Inductor bir xil sabab bilan to‘xtadi:
`BackendCompilerFailed: Cannot find a working triton installation.` Bu model
parity yoki sifat muammosi emas, lokal toolchain cheklovi. Compiler o‘rnatilishi
yoki default muhit avtomatik o‘zgartirilmaydi.

**Keyingi yo‘l:** Tritonga bog‘liq bo‘lmagan static-index/custom fused decode
path. **Batafsil:** `results/P003_NATIVE_COMPILE_AUDIT_20260910.md`.

### C-RUNTIME-NATIVE-CUDAGRAPH-001 — Dynamic-width route partition is not graph-safe

**Status:** `OPEN BLOCKER — FIXED FALLBACK ONLY`
**Muammo:** Runtime track / P-003

Fixed K=16 va learned-width batch=1 fallback CUDA Graph capture’da exact
parity bilan ishladi, lekin graph/eager `1.006x/1.001x` bo‘lib material tezlik
bermadi. Learned dynamic K=8/K=16 batch=32 capture vaqtida route-dependent
GPU shartlari va o‘zgaruvchan narrow/wide subsetlar sabab `operation not
permitted when stream is capturing` xatosiga uchradi. Bir inputdan olingan
route partitionni keyingi inputga graph orqali ko‘chirish xavfsiz emas.

**Keyingi yo‘l:** stable dispatch shape yoki route-dependent outputni saqlovchi
custom static-index kernel. Dynamic route’ni hozircha eager/opt-in qoldirish.
**Batafsil:** `results/P003_NATIVE_CUDA_GRAPH_AUDIT_20260910.md`.

### C-RUNTIME-NATIVE-FUSED-001 — Factorized native dispatch overhead

**Status:** `PROMISING OPT-IN — SHAPE-CACHE/CONCURRENCY SMOKE VALIDATED; PRODUCTION INTEGRATION OPEN`
**Muammo:** Runtime track / P-003 / P-007

Ordered factorized-additive 500M bank uchun inference-only custom CUDA kernel
qo‘shildi. Uch seedli 480-example timingda fixed K=8 `37.20 → 22.03 ms`
(`−40.8%`), fixed K=16 `58.38 → 35.32 ms` (`−39.5%`), learned K=8/16
`41.32 → 30.90 ms` (`−25.2%`) bo‘ldi. 960-example timingda mos ravishda
`−47.2%`, `−45.6%`, `−26.4%` chiqdi. PyTorch reference bilan maksimal logit
farqi `7.63e-6`; route va active width o‘zgarmadi. Fixed K=16 fused kernel
batch 1/32 CUDA Graph capture’dan ham parity bilan o‘tdi.

Kernel faqat ordered two-slot additive factor bankni qamraydi; pair/product,
hidden-gate, address-residual, serial va backward yo‘llari avtomatik ravishda
PyTorch fallbackda qoladi. Shuning uchun default almashtirilmadi.

96-batch-per-condition OOD quality control’da fused learned variant va PyTorch
learned controlning exact/hard/active-width metrikalari uch seedda bir xil
chiqdi; maksimal CE farqi `1.12e-8`. Demak hozirgi kernelning floating-point
accumulation farqi ko‘rilgan recurrent route yoki sifatni o‘zgartirmagan.

Keyingi uch seedli batch-shape sweep B=15/120/240/480/960 da fixed K=16
uchun torch→fused o‘rtacha `7.52→7.25`, `23.17→11.35`, `38.75→17.91`,
`65.48→29.76`, `116.43→45.31 ms` berdi (`−3.5%` dan `−61.1%` gacha).
Learned K=8/16 uchun mos yutuq `−22.5%/−21.4%/−33.7%/−34.5%/−38.8%` bo‘ldi.
Har bir shape’da maksimal parity xatosi `5.72e-6` dan oshmadi. B=15 da
`dynamic_width_min_batch=32` guard sabab learned yo‘l to‘liq K=16 ishlatdi;
bu majburlangan active-path emas, serving overheadini himoyalovchi mavjud guard.

Unsupported feature fallback uchun 9 CUDA test o‘tdi: unordered slots, shared
mix, query mix, pair/product/hidden-product/hidden-gate, serial composition va
address residual holatlarida native extension chaqirilmaydi, PyTorch yo‘li
ishlaydi. Sequence `6/8/16/32` sweepida B=120 uchun fixed K=16 yutug‘i
`40.7%/48.9%/50.2%/50.9%`, learned yutug‘i esa `29.5%/29.6%/30.7%/27.9%`
bo‘ldi; maksimal parity xatosi `5.72e-6`. Shu sabab batch/sequence/fallback
bosqichi yopildi. B=120, seq=6/32 shape’larini navbatlab va ikki CUDA
stream’da parallel ishlatgan serving smoke’da ham maksimal parity `5.72e-6`
bo‘ldi; shape/state aralashuvi kuzatilmadi. Bu dastlabki smoke shape-cache va
HTTP serverni qamramagan edi; keyingi sinovlar quyida alohida qayd etilgan.

`NativeFusedShapeCache` opt-in calleri bounded LRU CUDA Graph cache, shape
`(batch, sequence, stream)` key, dynamic-width eager fallback va capture-failure
fallback bilan qo‘shildi. Seed17 500M real checkpointida B=1 seq=6/32 uchun
2 capture/52 hit/fallback 0, cached `2.10/2.17 ms` bo‘ldi; B=8 uchun
`2.42/2.55 ms`, maksimal graph-eager parity `1.91e-6`. Unit testlar eviction,
dynamic fallback va capture failure fallbackni ham qamradi. Bu production
caller smoke’ni yopadi. Multi-worker’da cache ownership per-process ekanligi
va bir xil CUDA streamdagi requestlar lock bilan serialize qilinishi keyingi
smoke’da tasdiqlandi.
Same-stream policy 500M real checkpointda 4 worker/16 request bilan sinovdan
o‘tdi: 1 capture/16 hit, barcha outputlar reference bilan mos, maksimal xato
`1.91e-6`. Per-entry lock bir streamdagi callerlarni xavfsiz serialize qiladi;
cross-process reuse esa bloklanadi va har worker model/cache’ni o‘zi yaratishi
kerak.

`serve_native.py` va `NativeFusedService` orqali haqiqiy threaded HTTP entry
point qo‘shildi. `/health`, `/stats` va `/infer` endpointlari input shape,
sequence limit, token range va process ownershipni tekshiradi. Real seed17
500M checkpointida B=1/B=8 va seq=6/32 uchun 4 shape bir martadan capture va
keyin bir martadan cache hit berdi: `4 capture / 4 hit / 0 eager fallback`;
predictionlar takroriy so‘rovlarda mos, eager bilan maksimal logit farqi
`1.91e-6`. Birinchi HTTP chaqiriqlar `91.6–124.7 ms`, takroriy chaqiriqlar
`9.3–24.9 ms` bo‘ldi. Local server-entry smoke yopildi, ammo TLS,
authentication, batching/admission control, process supervision va production
multi-process launcher hali ochiq.

Mustaqil uzoq quality control’da fused va torch backendlari uch seed/to‘rt
condition bo‘yicha exact accuracy’da bir xil chiqdi, maksimal CE farqi
`1.61e-8`. Seed19 uniform exact `66.61%` va hard mean `31.76%` bilan seed17/18
dan ancha past, lekin torch control ham aynan shu raqamlarni berdi. Bu fused
kernel regressiyasi emas, checkpoint/training seed barqarorligi alohida
muammo ekanini ko‘rsatadi.

**Keyingi tajriba:** deployment-specific launcherda har worker uchun model va
cache lifecycle’ni mustaqil yaratish, keyin admission/batching siyosatini
stress-test qilish. **Batafsil:**
`results/P003_NATIVE_FUSED_DISPATCH_AUDIT_20260910.md`.

500M bankda `routing_capacity=22800` va `routing_depth=5` clamp qilinadigan
control full 500Mga nisbatan uniformda `+0.495 pp`, hard-taskda `+1.259 pp`
berdi. Bu route fragmentation haqiqiy omil ekanini ko‘rsatadi, lekin clamp
300M reference’dan uniformda `−0.634 pp` va hard-taskda `−0.933 pp` pastligicha
qoldi. Demak qo‘shimcha saqlangan qatorlarni shunchaki yashirish scalingni
to‘liq hal qilmaydi.

**Batafsil:** `results/P003_NATIVE_CAPACITY_CLAMP_AUDIT_20260910.md`.

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

Qwen one-token grouped dispatch uchun `single_token_fast_path` opt-in yo‘li
qo‘shildi. Isolated `[1,1,1024]` layerda token-loopga nisbatan `2.230 →
0.365 ms`; sakkizta layer full-forward smoke’da eski grouped `36.294 →
32.560 ms` (`10.3%`) bo‘ldi. Final logits max farqi `7.629e-6`, lekin sparse
fast path parentdan hali `1.095x` bo‘lib qoldi. Shu sabab bu dispatch foydali
optimallashtirish, ammo one-token muammosi yechildi deb hisoblanmadi va
trained quality auditgacha defaultga olinmadi. Batafsil:
`results/RUNTIME_QWEN_SINGLE_TOKEN_FAST_PATH_20260908.md`.

`grouped-fused` va `packed` nazoratlari ham foyda bermadi: isolated timinglar
mos ravishda `0.713 ms` va `3.297 ms`, sakkiz qatlamli grouped-fused full
forward esa `36.725 ms` bo‘ldi. Full Qwen profileri shu smoke’da
scaled-dot-product attentionning o‘zi taxminan `25.5 ms` CUDA vaqt olganini
ko‘rsatdi. Demak Qwen FFN dispatchini tezlatish butun Transformer one-token
yo‘lini avtomatik ravishda tezlashtirmaydi; bu Native Engine’ning
attention-free yo‘liga qarshi dalil emas.

Router-only ablation ham muammo faqat router emasligini ko‘rsatdi: sakkiz
qatlamli one-token smoke’da odatiy router `37.245 ms`, statik nol-logit
controller `33.822 ms`, dense parent `29.019 ms` bo‘ldi. Router taxminan
`3.423 ms` tejaydi, lekin router olib tashlanganda ham sparse yo‘l `1.166x`
sekin. Shuning uchun keyingi haqiqiy runtime sakrashi fused
router+top-k+dispatch kernelidan kelishi kerak; oddiy router MLPni
kichraytirishning o‘zi yetarli emas. Batafsil:
`results/RUNTIME_QWEN_ROUTER_OVERHEAD_20260908.md`.

**V0.195 fixed-shape CUDA Graph replay (2026-09-09):** Qwen3-0.6B sakkiz
qatlamli `batch=1, sequence=1` smoke’da K=5 graph replay `13.730 ms` va
`0.467x` dense parent, K=6 esa `17.087 ms` va `0.588x` parent bo‘ldi; eager
sparse yo‘l mos ravishda `31.919/33.963 ms` edi. Graph/eager max logit farqi
`1.6e-5` ichida qoldi. Bu fixed-shape launch overheadi katta ekanini va
runtime uchun kuchli yangi signal borligini ko‘rsatadi. Router/circuit
matematikasi o‘zgarmadi, child’lar o‘qitilmagan runtime smoke bo‘lgani uchun
quality claim emas. `ACCEPTED OPT-IN`; trained K=5/K=6, `use_cache`, input
buffer update (fixed-shape `use_cache=False` parity o‘tdi) va dynamic shape
auditlari hali ochiq. Generic Transformers `StaticCache` bilan `use_cache=True`
graph screen parent-only nazoratda ham max capture/replay xatosi `12.78/1.74`
berdi va `UNSAFE` deb rad qilindi. **V0.196 custom fixed-KV replay** esa
`use_cache=True` bilan K=5’da 200 iteratsiyada `14.703 ms` (`0.516x` dense
parent), K=6’da `17.084 ms` (`0.595x`) berdi; replay max logit xatosi
`1.22e-5`, alternate-token xatosi `9.66e-6` ichida qoldi. Bu custom cache
yo‘li uchun `POSITIVE OPT-IN`, lekin child’lar runtime smoke uchun
o‘qitilmagan; trained quality, prefill, `generate()`, dynamic shape va
defaultga olish hali ochiq. Batafsil:
`results/RUNTIME_QWEN_CUSTOM_KV_CUDA_GRAPH.md`.

**V0.197 trained custom-KV result (2026-09-09):** accepted K=5 recipe bilan
8 qatlam qayta o‘qitildi: teacher CE `4.785308`, same-seed sparse CE deltalari
`+0.036528` va repeat’da `+0.043563`, ikkalasi ham `+0.05` quality gate’dan
o‘tdi. Trained `use_cache=True` custom fixed-KV graph repeat’da `17.391 ms`,
dense parent `26.648 ms` (`0.653x`), sparse eager `31.496 ms`
(`graph/eager=0.552x`) bo‘ldi. Replay-vs-eager max logit error `7.63e-6`,
alternate-token error `5.49e-6`; `PARITY_PASS`. Shu bilan trained
fixed-shape runtime proof point **SOLVED/ACCEPTED OPT-IN** bo‘ldi. Bu umumiy
serving muammosi tugadi degani emas: prefill-to-decode, `generate()`, dynamic
shape, uzoq timing va production stream/kernel safety hali ochiq. Shu trained
cascade bilan 13-token promptdan 8-token greedy generation ham graph/eager
orasida exact match berdi; shape pool `1 capture / 1 hit` qayd etdi.
Trained batch follow-up’da graph/eager parity barcha batch’da `1.1e-5`dan
kichik qoldi, lekin dense-parent ratio batch 1/2’da `0.737x/0.936x`, batch
4/8’da `1.109x/1.445x` bo‘ldi. Demak trained correction katta batch’da
graph-safe bo‘lsa ham dense’dan tez emas; bu performance muammosi hali ochiq.
**Batafsil:** `results/RUNTIME_QWEN_TRAINED_CUSTOM_KV_GRAPH_AUDIT_20260909.md`.

**V0.198 multi-step cache-state result (2026-09-09):** bir prefix prefill’dan
keyin to‘rtta ketma-ket one-token graph replay’da token buffer va KV decode
pozitsiyasi har qadam yangilandi. To‘rtta qadam bo‘yicha maksimum
graph/eager logit xatosi `5.72e-6`, `PARITY_PASS`. Fixed-shape multi-step
cache-state muammosi **SOLVED/ACCEPTED OPT-IN**. Qolgan muammo — buni
`generate()` adapteriga ulash, dynamic shape uchun xavfsiz eager fallback,
shape-cache siyosati va production kernel/stream validation.
**Batafsil:** `results/RUNTIME_QWEN_CUSTOM_KV_MULTISTEP_20260909.md`.

**V0.199 generation-adapter result (2026-09-09):** fixed-shape greedy
generation helper prefix prefill’dan keyin 8 ta tokenni graph va eager
yo‘llarda bir xil chiqardi (`exact_token_match=true`, `PARITY_PASS`). Bu
runtime state loop muammosini **SOLVED/ACCEPTED OPT-IN** qiladi. Child’lar
ushbu smoke’da o‘qitilmagan; trained-child generation V0.197’da alohida
o‘tdi. Dynamic-shape handling va Hugging Face `generate()` bilan to‘liq
integratsiya hali ochiq. Qo‘shimcha control’da uncaptured budget uchun
`capture_on_miss=False` eager fallback mustaqil eager run bilan aynan mos keldi.
Bounded pool’da 3 ta shape capture qilinib, 1 ta shape eviction’dan keyin
eager fallback ham aynan mos keldi.
V0.200’da prefix uzunliklari 4 va 8 uchun alohida capture entry’lar yaratildi;
ikkalasi ham eager bilan exact match berdi va 4-token entry qayta ishlatildi.
**Batafsil:** `results/RUNTIME_QWEN_FIXED_GRAPH_GENERATION_20260909.md`.
**V0.200 batafsil:** `results/RUNTIME_QWEN_FIXED_GRAPH_PREFIX_SHAPES_20260909.md`.
**V0.201 batch-2 result (2026-09-09):** fast-path guard bitta flattened
tokenni emas, `sequence_length=1`ni tekshiradigan qilib tuzatildi. Batch=2’da
graph/eager generation batch 2/4/8’da exact match berdi; graph/eager ratios
`0.534x/0.594x/0.601x` bo‘ldi (`156.52/189.64/261.79 ms` graph va
`293.34/319.06/435.81 ms` eager). Har birida `1 capture / 13 hits` qayd
etildi. Batch 2–8 fixed-shape muammosi **SOLVED/ACCEPTED OPT-IN**. Batch>8,
trained batch quality, concurrency va production stream isolation hali ochiq.
**Batafsil:** `results/RUNTIME_QWEN_FIXED_GRAPH_BATCH_SHAPE_20260909.md`.
**V0.202 correction backend result (2026-09-09):** trained B8’da vectorized
rank-64 correction graph-safe bo‘ldi, lekin graph/eager `1.091x`; packed
accumulation eager’da `79.794 ms` bo‘ldi va capture vaqtida device
`torch.where` sabab yiqildi. Packed backend graph uchun **REJECTED**,
vectorized hozirgi safe backend sifatida qoldi. Katta batch’da dense’dan
tezroq bo‘ladigan static-index/fused correction kernel hali ochiq.
**Batafsil:** `results/RUNTIME_QWEN_TRAINED_CORRECTION_BACKEND_AUDIT_20260909.md`.
**V0.203 BMM result (2026-09-09):** single-token rank-64 correction uchun
explicit BMM qo‘llanganda trained B8 graph/eager `1.091x → 1.034x` bo‘ldi,
parity xatosi `8.94e-6`, quality CE delta `+0.035745`. Bu kichik ijobiy
optimallashtirish, dense B8 latency muammosining to‘liq yechimi emas.
**Batafsil:** `results/RUNTIME_QWEN_TRAINED_CORRECTION_BMM_AUDIT_20260909.md`.
**V0.204 rank natijasi (2026-09-09):** rank32 correction trained K=5 bilan
seed2026/17’da quality CE delta `+0.031617/+0.039860` va B8 graph/dense
`1.411x/1.344x` berdi; parity `1e-5` ichida. Rank16 quality gate’dan o‘tsa ham
B8 `1.479x`, ya’ni rank32’dan yaxshiroq emas. Rank32 **ACCEPTED OPT-IN**,
default rank64 saqlandi; uzoqroq multi-seed audit va fused/static-index kernel
hali ochiq.
**Batafsil:** `results/RUNTIME_QWEN_CORRECTION_RANK_AUDIT_20260909.md`.
**V0.205 rank8 natijasi (2026-09-09):** 40 warmup/100 iteratsiyali uzun
o‘lchovda seed2026/17 uchun B8 graph/dense `1.390x/1.008x`,
graph/sparse-eager `0.924x/0.965x`, quality CE delta `+0.024122/+0.029971`
bo‘ldi; parity `1e-5` ichida. Rank8 **ACCEPTED OPT-IN**, hozirgi eng tez
candidate; default rank64 saqlandi. GPU timing variance va fused/static-index
correction kernel hali ochiq muammo.
**Batafsil:** `results/RUNTIME_QWEN_CORRECTION_RANK8_AUDIT_20260909.md`.
**V0.206 base BMM natijasi (2026-09-09):** single-token base projectionda
BMM eager `37.238 ms`, graph `35.494 ms`; einsum eager `37.626 ms`, graph
`34.974 ms`. BMM parity-safe bo‘lsa ham graph’da tezlashtirmadi, shuning uchun
**REJECTED AS DEFAULT**, einsum saqlandi. Packed capture probe’dan keyingi
CUDA context muammosi ham audit tartibini o‘zgartirib tuzatildi.
**Batafsil:** `results/RUNTIME_QWEN_SINGLE_TOKEN_PROJECTION_AUDIT_20260909.md`.
**V0.207 Inductor muammosi (2026-09-09):** fused-child probe model parity
emas, lokal PyTorch `2.6.0+cu124` muhitida ishlaydigan Triton topilmagani
uchun bloklandi. Oddiy eager/graph yo‘l `PARITY_PASS`; Inductor **ENVIRONMENT-
BLOCKED**, default o‘zgarmadi. Keyingi yo‘l Inductorga bog‘liq bo‘lmagan
static-index/fused correction kernel.
**Batafsil:** `results/RUNTIME_QWEN_INDUCTOR_PROBE_20260909.md`.
**V0.208 correction ablation (2026-09-09):** correctionni butunlay olib
tashlash (`rank=0`) B8 graph/dense `1.047x` va graph/sparse-eager `0.898x`
berdi, lekin quality CE delta `+0.091023` bo‘lib `+0.05` gate’dan yiqildi.
Demak correction capacity hozirgi K=5 quality uchun **ZARUR**; rank8 eng
kichik viable candidate sifatida qoldi. Rank0 train bugi ham tuzatildi.
**Batafsil:** `results/RUNTIME_QWEN_CORRECTION_ABLATION_20260909.md`.
**V0.209 rank4 natijasi (2026-09-09):** correction rank4 uch seedda ham
quality gate’dan o‘tdi: CE delta `+0.037330/+0.040037/+0.040855`; B8
graph/dense `1.339x/1.330x/1.337x`, graph/sparse-eager
`0.887x/0.927x/0.899x`, parity `1.05e-5/8.58e-6/9.06e-6`. Generation
graph/eager va reused-shape parity uchala seedda ham exact. Rank4 hozirgi eng
kichik **ACCEPTED OPT-IN** candidate; default rank64 saqlandi. Uzunroq
training budget va fused/static-index correction kernel hali ochiq.
**Batafsil:** `results/RUNTIME_QWEN_CORRECTION_RANK4_AUDIT_20260909.md`.
**V0.210 long-budget natijasi (2026-09-09):** rank4 uchun child/hard qadamlar
`300→600`, router `100→200` qilindi. Seed2026/17 CE delta
`+0.027947/+0.038252` bo‘lib gate ichida qoldi; B8 graph/dense
`1.280x/1.265x`, graph/sparse-eager `0.904x/0.873x`, parity `9.54e-6/9.06e-6`.
Generation parity ikkala seedda exact. Qisqa budgetga xos tasodifiy signal
ehtimoli kamaydi, lekin rank4 defaultga ko‘tarilmadi; fused/static-index
correction kernel hali asosiy runtime muammosi.
**Batafsil:** `results/RUNTIME_QWEN_CORRECTION_LONG_BUDGET_20260909.md`.
**V0.211 custom CUDA correction (2026-09-09):** tanlangan K=5 correction uchun
fixed-shape CUDA kernel qo‘shildi. Dastlab default stream bugi graph capture’ni
bo‘sh qoldirdi; current streamga o‘tkazilgach graph/eager parity `9.3e-6/9.5e-6`
bo‘ldi. Trained B8 A/B’da vectorized→CUDA kernel eager `44.48→44.78 ms`, graph
`49.36→48.88 ms`; demak katta tezlik sakrashi yo‘q, **DEFAULT EMAS, OPT-IN**.
Quality va model body o‘zgarmadi; keyingi katta yutuq uchun fused base+correction
dispatch yoki kernel fusion kerak.
**Batafsil:** `results/RUNTIME_QWEN_CORRECTION_CUDA_KERNEL_AUDIT_20260909.md`.
**V0.212 effective-output folding (2026-09-09):** correctionni
`W_eff = W_out + mix_out·mix_in·W_out` ga oldindan birlashtirish local formula
va CPU paritydan o‘tdi. Ammo barcha 8 layer cascade’da vectorized final logitsga
nisbatan max error `1.45` chiqdi; kichik floating farqlar keyingi layer routingini
o‘zgartirdi. **REJECTED FOR ADOPTION**; default va quality yo‘li o‘zgarmadi.
Bu yo‘l faqat route freeze qilingan yoki butun layer fused qilingan holatda qayta
ko‘rilishi mumkin.
**Batafsil:** `results/RUNTIME_QWEN_EFFECTIVE_OUTPUT_FOLDING_AUDIT_20260909.md`.
**V0.213 rank4 third long-budget seed (2026-09-09):** seed42 ham bir xil
`600/600/200` recipe bilan quality gate’dan o‘tdi: CE delta `+0.040236`,
top-1 agreement `0.7976`, B8 graph/dense `1.234x`, graph/sparse-eager
`0.871x`, final-logit parity `8.34e-6`, generation parity exact. Shu bilan
rank4 long-budget natijasi uch seedda takrorlandi; **ACCEPTED OPT-IN** dalili
kuchaydi, ammo default rank64 saqlandi va katta runtime sakrashi hali yo‘q.
**Batafsil:** `results/RUNTIME_QWEN_CORRECTION_LONG_BUDGET_20260909.md`.
**V0.214 fused base+correction dispatch (2026-09-09):** yangi fixed-shape
CUDA kernel selected Qwen output va rank4 correctionni bitta dispatchda
hisoblaydi. Seed42/2026’da vectorized backendga nisbatan B8 graph
`0.842x/0.859x`, final-logit farqi `1.62e-5/1.67e-5`; dense parentga nisbatan
`0.957x/1.000x` bo‘ldi. **ACCEPTED OPT-IN**, default vectorized saqlandi:
bu correction overheadini kamaytiradi, lekin router/top-k/attention va
dynamic shape serving muammolari ochiq qoladi.
**Batafsil:** `results/RUNTIME_QWEN_FULL_CORRECTION_DISPATCH_AUDIT_20260909.md`.
**V0.215 fused correction batch sweep (2026-09-09):** batch8’da full kernel
ikkala seedda ham vectorizeddan tez (`0.862x/0.495x`), parity xatosi
`1.53e-5/1.29e-5`. Batch1’da esa natija qarama-qarshi (`1.276x/0.720x`),
shuning uchun avtomatik batch-size switch kiritilmadi; vectorized default
saqlandi. Bu muammo **OPEN**: repeated interleaved timing va router/top-k
fusion kerak.
**Batafsil:** `results/RUNTIME_QWEN_FULL_CORRECTION_BATCH_SWEEP_AUDIT_20260909.md`.
**V0.217 interleaved correction timing (2026-09-09):** ikkala seedda ham
batch1 fused/vectorized `1.478x/1.468x` sekin, batch8 esa faqat
`0.978x/0.975x` tezroq chiqdi; final-logit xatosi `8.58e-6–1.53e-5`.
Shuning uchun fused-fullning katta speed claim’i **REJECTED**, parity-safe
opt-in kodi saqlandi, vectorized default o‘zgarmadi. Keyingi katta target —
router+top-k+selected dispatch+correctionni bir kernelga birlashtirish.
**Batafsil:** `results/RUNTIME_QWEN_FULL_CORRECTION_INTERLEAVED_AUDIT_20260909.md`.
**V0.218 token-block correction (2026-09-09):** atomicAdd’ni olib tashlash
uchun bitta token ichidagi K=5 groupni bitta blockda ketma-ket hisoblovchi
kernel sinovdan o‘tdi. Parity yaxshi (`9.54e-6/1.62e-5`), ammo vectorizedga
nisbatan B1 `4.661x`, B8 `1.726x` sekin. Atomics asosiy bottleneck emas,
serial FFN hisoblash parallel GEMMdan yutqazadi; **REJECTED FOR SPEED**.
**Batafsil:** `results/RUNTIME_QWEN_TOKEN_BLOCK_CORRECTION_AUDIT_20260909.md`.
**V0.219 router/top-k/dispatch stage profile (2026-09-09):** fixed hidden
state o‘lchovida sakkiz child uchun B1 router+top-k `1.391 ms`, selected FFN
dispatch `2.553 ms`, full child `3.578 ms`; B8’da mos ravishda `1.008 ms`,
`17.925 ms`, `18.236 ms`; B32’da `1.180 ms`, `70.951 ms`, `71.547 ms` bo‘ldi.
Demak B8+da router emas, selected FFN dispatch asosiy xarajat; fusionning
end-to-end ceiling’i cheklangan. **DIAGNOSTIC BASELINE**.
**Batafsil:** `results/RUNTIME_QWEN_ROUTER_TOPK_DISPATCH_PROFILE_20260909.md`.
**V0.220 fused router (2026-09-09):** ikkita linear+SiLU router, top-k va
softmax bitta opt-in CUDA kernelga yig‘ildi. Tied bo‘lmagan deterministic
routerda route-stage B1 `0.636x`, B8 `0.675x`, B32 `1.085x` bo‘ldi; child
forward B1 `0.913x`, B8 `1.024x`, B32 `1.014x`; full-model one-token smoke esa
B1 `1.000x`, B8 `1.020x` bo‘ldi. ID mismatch `0`, child output max xatosi
`4.77e-7/9.54e-7/1.91e-6`, full-model max xatosi `3.81e-6/9.54e-6`.
B1’da kichik ijobiy mikro-optimallashtirish bor, lekin universal yoki
end-to-end speedup emas; **ACCEPTED OPT-IN MICRO-OPTIMIZATION, DEFAULT
UNCHANGED**. Tied fresh routerlar uchun PyTorch GPU top-k tartibi aniq route
contract sifatida belgilanmagani sabab custom kernel faqat auditdan o‘tgan
trained/non-tied routerda ishlatiladi.
**Batafsil:** `results/RUNTIME_QWEN_FUSED_ROUTER_AUDIT_20260909.md`.
**V0.222 trained fused subset-router (2026-09-09):** real accepted K=5
`subset-router` va 56 subset bilan `300/300/100`, rank64 seed2026 qayta
tekshirildi. Sifat `+0.042135` CE delta bo‘lib gate ichida qoldi. Fused subset
route eager B1/B8’da `0.980x/1.020x`, graphda `1.032x/1.011x`; fused-vs-Torch
eager max logit farqi `5.72e-6/1.07e-5`. Dastlab current-stream xatosi graphda
invalid ID chiqardi, `getCurrentCUDAStream()` tuzatuvidan keyin graph parity
`6.20e-6/1.10e-5` bilan o‘tdi. 8-token CUDA-Graph generation exact token
match berdi. **PARITY-SAFE OPT-IN, DEFAULT UNCHANGED**; router fusion umumiy
graph serving sakrashi bermadi, selected FFN dispatch asosiy keyingi target
bo‘lib qoldi.
**Batafsil:** `results/RUNTIME_QWEN_FUSED_SUBSET_ROUTER_TRAINED_AUDIT_20260909.md`.
**V0.223 graph-safe grouped selected-FFN (2026-09-09):** grouped dispatchdagi
`torch.bincount` va host-side dynamic `max_count` CUDA Graph capture’ni
to‘sardi. Capture vaqtida xavfsiz token-count upper bound qo‘llandi, eager
prefillda esa dynamic bound saqlandi. Trained K=5 seed2026/2027’da grouped /
single-token graph ratio B1 `0.963x/0.953x`, B8 `0.589x/0.589x`; B32 extension
`0.423x`. Max logit xatosi `1.4e-5` dan kichik, 8-token generation exact
match. Prefix 4/32/128 da B8 graph ratios `0.593x/0.600x/0.663x` bo‘lib,
uzun contextda ham signal saqlanadi. Bu router fusiondan farqli ravishda
selected FFN bottleneckga tegadigan eng kuchli runtime signal. **PROMISING;
serving policy ochiq. Lekin dense parent bilan grouped sparse graph hali
B1/B8/B32’da `1.099x/1.146x/1.259x`, ya’ni dense’dan sekin; production-shape
validation va selected-FFN optimizatsiyasi ochiq.**
**Batafsil:** `results/RUNTIME_QWEN_GROUPED_GRAPH_SAFE_AUDIT_20260909.md`.
**V0.224 grouped-fused selected FFN (2026-09-09):** gate/value projectionni
bitta BMMga birlashtirgan variant graphda oddiy groupedga nisbatan faqat
B1/B8/B32’da `1.000x/0.993x/0.997x` bo‘ldi; eager B1/B8 foydali, B32 neytral.
Generation exact parity saqlandi, lekin dense parentga nisbatan hali
`1.108x/1.136x/1.257x` sekin. **PARITY-SAFE MICRO-OPTIMIZATION, DEFAULT
UNCHANGED**; katta keyingi target selected FFN launch/packing overheadi.
**Batafsil:** `results/RUNTIME_QWEN_GROUPED_FUSED_AUDIT_20260909.md`.

**V0.229 rank-1 long-budget audit (2026-09-09):** correction ranki `1`ga
tushirilgan trained K=5 grouped-fused yo‘l child/hard/router `600/600/200`
protokolida seed2026/17 bilan qayta o‘lchandi. CE delta `+0.045878/+0.035236`
bo‘lib, ikki seedda ham `+0.05` quality gate’dan o‘tdi; mean delta `+0.040557`.
B8 CUDA Graph’da grouped-fused/dense `1.048x/1.044x`, grouped-fused/grouped
`1.000x/0.997x` bo‘ldi. Ikkala seedda 8-token generation exact parity va
`1.1e-5`dan kichik logit parity saqlandi. Rank-1 hozirgi eng kichik viable
opt-in runtime candidate, lekin dense parentdan hali 4–5% sekin; rank64
default o‘zgarmadi. Keyingi asosiy ish rankni yana qisqartirish emas, selected
FFN launch/packing overheadini static-index yoki to‘liq fused dispatch bilan
olib tashlash.
**Batafsil:** `results/RUNTIME_QWEN_RANK1_LONG_AUDIT_20260909.md`.

**V0.230 rank-1 shape sweep (2026-09-09):** rank-1 trained K=5 yo‘l B1,
B8 va B32 fixed-shape graph’da ikki seed bilan tekshirildi. Dense-parentga
nisbiy grouped-fused vaqtlar seed2026 uchun `1.080x/1.044x/1.065x`, seed17
uchun `1.078x/1.042x/1.060x` bo‘ldi. Quality CE delta `+0.047808/+0.042103`
bo‘lib gate ichida qoldi; generation parity barcha shape’larda exact. Demak
rank-1 B8ga xos emas va opt-in candidate sifatida saqlanadi, lekin hech bir
shape’da dense’dan tez emas. Grouped-fused oddiy groupeddan barqaror ustun
emas; batch policy kiritilmadi. Keyingi ish router yoki rankni qisqartirish
emas, selected FFN launch/packing overheadini static-index/tiled yoki full
fused implementation bilan kamaytirish.
**Batafsil:** `results/RUNTIME_QWEN_RANK1_SHAPE_SWEEP_20260909.md`.

**V0.231 grouped metadata cache (2026-09-09):** route-independent
`arange/token/slot` metadata fixed-shape cache qilindi va ikki seedda B1/B8/B32
tekshirildi. Cached/grouped graph ratio seed2026 uchun `0.996x/1.000x/1.002x`,
seed17 uchun `0.997x/0.994x/0.996x`; generation va logit parity exact. Bu
material speedup emas, lekin xavfsiz opt-in micro-optimization sifatida
saqlandi. Route metadata qayta yaratish keyingi asosiy bottleneck emas; selected
FFN packing/projection/scatter uchun tiled yoki full fused kernel kerak.
**Batafsil:** `results/RUNTIME_QWEN_GROUPED_CACHED_METADATA_AUDIT_20260909.md`.

**V0.232 grouped correction fusion (2026-09-09):** selected FFN outputni
wrapperga qayta joylashtirib correctionni alohida hisoblash o‘rniga, correction
grouped accumulation ichiga qo‘shildi. Trained K=5 rank-1 seed2026/17da
grouped baselinega nisbatan B1/B8/B32 ratio mos ravishda `0.997x/0.988x/0.992x`
va `0.991x/0.989x/0.993x` bo‘ldi. Dense parentga nisbatan hali
`1.081x/1.039x/1.066x` va `1.072x/1.037x/1.061x`; quality delta
`+0.045226/+0.040347`, generation parity exact. Demak kichik, izchil
micro-optimization bor, lekin katta speedup yo‘q; backend faqat opt-in, keyingi
target packing/projection/accumulationni tiled yoki full fused kernelda birlashtirish.
**Batafsil:** `results/RUNTIME_QWEN_GROUPED_CORRECTION_FUSED_AUDIT_20260909.md`.

**V0.233 uniform K-subset accumulation (2026-09-09):** subset-router’dagi
uniform `1/K` weight va `hard_route_scale=K` cancellationidan foydalanadigan
shortcut sinab ko‘rildi. Uniform/correction-fused graph ratio seed2026 uchun
B1/B8/B32 `0.996x/0.998x/1.000x`, seed17 uchun `1.002x/0.996x/0.995x` bo‘ldi;
generation parity exact. Foyda kichik va shape/seed bo‘yicha izchil emas, shu
sabab **REJECTED FOR ADOPTION**, faqat exact subset-router uchun diagnostik
opt-in saqlandi. Keyingi target hanuz selected FFN packing/projection/scatter
fusion.
**Batafsil:** `results/RUNTIME_QWEN_GROUPED_UNIFORM_ACCUM_AUDIT_20260909.md`.
Batafsil:
`results/RUNTIME_QWEN_CUDA_GRAPH.md`.

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

Kengaytirilgan state-path diagnostikasi va scale sweep 2026-09-09 kuni
100M/300M/500M, seed17/18 checkpointlarda bajarildi. Correction delta route
almashtirilganda haqiqatan o‘zgaradi (`~1.19–1.30x` natural delta), shuning
uchun circuit output state yo‘liga butunlay ulanmagan degan gipoteza rad etildi.
Lekin seed17dagi 100M/300M/500M diagnostikasida correctionni `scale=0` qilish
CE’ni `−0.00865/−0.00472/−0.00622` ga yaxshiladi; 300M/500M hard accuracy
`−0.42 pp` bo‘ldi. Demak correction foydali computationni ham olib keladi,
ammo uning final lossga yo‘nalishi beqaror.

Olti checkpointli inference-only sweepda mean natural CE scale `0/0.1/0.25/
0.5/1.0` uchun `0.428571/0.428466/0.428340/0.428509/0.430237` chiqdi.
Scale `0.25`ning `−0.00023` aggregate CE farqi turli seed/modelda universal
emas; har checkpoint optimal scale’i turlicha. `scale=0.25` yoki correctionni
butunlay o‘chirish default sifatida qabul qilinmadi.

**Yangi qaror:** P-007ni oddiy amplitude/router retrieval muammosi deb yopish
mumkin emas. Keyingi yo‘l correction contributionni final task loss bilan
bog‘laydigan opt-in training objective va bounded state/output interface;
capacity-only scaling va yangi statik scale sweep hozircha to‘xtatiladi.

**Batafsil:** `results/P007_STATE_PATH_SCALE_EXTENDED_AUDIT_20260909.md`.

2026-09-10 depth specialization diagnostikasi 10k checkpointlarda depth-2/3
selected route correctioni 500Mda 300Mdan kuchliroq emasligini ko‘rsatdi:
uniform probe’da route/query normasi depth-2 uchun `6.63% → 5.22%`, depth-3
uchun `6.40% → 4.47%` tushdi. Shu gipotezaga mos `circuit_delta_scale=2`
training arm ham tekshirildi: ikki seed, 3000 qadamda accuracy faqat `+0.169
pp`, hard-task mean `+0.033 pp`, CE esa `+0.00466` yomonlashdi. Universal
amplitude oshirish P-007ni yechmadi va qabul qilinmadi.

**Depth/amplitude audit:** `results/P007_NATIVE_DELTA2_DEPTH_AUDIT_20260910.md`.

Keyingi ikki inference-only pilot ham yakunlandi. Per-example correction
advantage aralash chiqdi (100M seed17/18 positive fraction `46.88%/52.08%`),
lekin advantage bilan correction normasi korrelyatsiyasi kuchsiz
(`−0.177/+0.031`). Calibration’da task-level scale tanlab, boshqa batchda
tekshirish natural scale=1ni yengmadi: CE delta `+0.00500/+0.00057`.

Final-loss advantage’dan feature gate fit qilish calibration targetini 100%
eslab qoldi, ammo evalda CE `+0.00535/+0.00125` yomonlashdi va seed18 accuracy
`−0.42 pp` tushdi. Shuning uchun task-conditioned scale ham, misol-darajasidagi
oddiy linear advantage gate ham **REJECTED**. P-007 endi universal scale/gate
muammosi emas; correction signalini circuit specialization va final output
training objective bilan barqaror align qilish muammosi.

**Batafsil:** `results/P007_ADVANTAGE_GATE_AUDIT_20260909.md`.

P-003 uchun “staged growthda inherited prefix drift qilyapti” gipotezasi ham
one-seed 2000-step paired pilotda tekshirildi. 1408 circuit/key prefixini
muzlatish unfrozen control bilan bir xil `80.47%` accuracy berdi, ammo CE
`0.580555` vs `0.575009` (`+0.00555`) bo‘ldi; training `+5.35%`, peak VRAM
`+760 MB`. Stable-prefix route coverage biroz yaxshilangan bo‘lsa ham quality
gate bermadi va to‘liq screen’ga kengaytirilmadi. Bu variant **REJECTED**;
P-003 capacity plateau’ni faqat prefix drift bilan tushuntirib bo‘lmaydi.

**Batafsil:** `results/P003_STABLE_PREFIX_GROWTH_AUDIT_20260909.md`.

Post-GRU correction residual (`post_correction_residual_scale=β`) ham
inference-only tekshirildi. `β=1.0` seed17’da `+1.25 pp`, seed18’da
`−0.42 pp` natural accuracy berdi; CE va route-replay sensitivity ham
qarama-qarshi bo‘ldi. Shuning uchun correction’ni GRU’dan keyin bevosita
qo‘shish mavjud checkpoint uchun barqaror quality fix emas va training/defaultga
qabul qilinmadi. Opt-in API saqlandi.

**Residual audit:** `results/P007_POST_CORRECTION_RESIDUAL_AUDIT.md`.

Gated state-write (`memory_write_mode=gated`) ham tekshirildi. 300M ordered
shared-route-key, ikki seed, 3000 qadamli screen’da mean accuracy
`68.451% → 67.969%` (`−0.482 pp`), depth-2/3 mean `33.366% → 33.203%`
(`−0.163 pp`) bo‘ldi; CE `0.99639 → 0.99445` yaxshilangan bo‘lsa ham hard
quality oshmadi. Demak generic state-preservation gate P-007ni yechmaydi va
P-005dagi CE/hard mismatch saqlanadi.

**Memory-write audit:** `results/P007_NATIVE_MEMORY_WRITE_AUDIT_20260910.md`.

Task tokenini routerga alohida embedding sifatida berishning uch varianti ham
tekshirildi. Embeddingni state update’ga ham qo‘shish ikki seedda mean
accuracy’ni `68.451% → 68.047%` tushirdi. Faqat routerga `1.0×` berish accuracy’ni
`+0.313 pp` va CE’ni `−0.01401` yaxshilagan bo‘lsa ham hard-task mean
`−0.163 pp`, dead traffic `+8.78 pp` bo‘ldi. `0.25×` router-only variantida
accuracy `−0.156 pp`, hard-task mean `−0.228 pp` va dead traffic `+3.41 pp`
bo‘ldi. Demak task context routingni task-specific qiladi, lekin depth-2/3
qualityni ishonchli oshirmaydi; scale tuning qabul qilinmadi.

**Task-context audit:** `results/P007_NATIVE_TASK_CONTEXT_AUDIT_20260910.md`.

Step-specific rank-8 circuit adapter 300M ikki seedli 3000-qadamli screen’da
mean accuracy’ni `68.451% → 69.063%` (`+0.612 pp`), hard-task mean’ni
`+1.302 pp` va CE’ni `−0.01671` yaxshiladi. 10000 qadamda 300M foydasi
`+0.352 pp`, 500M foydasi esa `+0.026 pp` bo‘ldi; hard-task mean ikkala
scale’da ham yaxshilandi. Edge auditda 300M low-edge `+0.625 pp`, high-edge
`−1.389 pp`; 500M low/high edge `−1.575/−1.047 pp` chiqdi. Demak bu hozirgi
eng kuchli native candidate, ammo high-edge robustness va 300→500 overall
scaling muammosi ochiq; defaultga olinmadi.

Adapterni faqat depth-2/3ga qo‘llash ikki seedda `68.555%` berdi va to‘liq
adapterdagi `69.063%` foydaning ko‘p qismini yo‘qotdi. Step-1 interface ham
zarur ekanini ko‘rsatadi.

**Step-adapter audit:** `results/P007_NATIVE_STEP_ADAPTER_SCALE_AUDIT_20260910.md`.

### C-P003-NATIVE-FACTORIZED-001 — Virtual factor bank does not turn address count into quality

**Status:** `REJECTED AS QUALITY/CAPACITY FIX; RETAINED OPT-IN FOR COMPRESSION/RUNTIME`
**Muammo:** P-003 / P-007

Native V0’da 7,552 virtual circuit addressini 87 reusable factor row orqali
ifodalash 1,000 qadamlik ikki-seed screen’da mustaqil 100M bankdan `−0.195 pp`
accuracy va `+0.05699` validation CE yomon chiqdi. Buning evaziga model
`100.47M → 3.00M` parametrga, training `72.33s → 36.71s` ga va peak VRAM
`1946 → 680 MB` ga tushdi. Factorized router o‘rtacha faqat `~876/7552`
virtual addressni ishlatdi, dead-address fraction `~88.4%` bo‘ldi. Rank-4
pair-basis qo‘shilishi ham plain factorized arm’dan `−0.69 pp` accuracy yomon
chiqdi.

Demak bu yo‘l hozircha sifat yoki sig‘im muammosini hal qilgani yo‘q; virtual
manzillar sonini ko‘paytirish o‘z-o‘zidan o‘rganilgan capacity bermayapti.
Compression/runtime opt-in sifatida saqlandi, default o‘zgarmadi. P-003 va
P-007 ochiq qoladi; keyingi ish factor-row exposure/specialization va route
collapse sababini o‘lchashi kerak. Virtual address dead fraction factor-row
dead fraction bilan bir xil emas; `train.py evaluate()` endi reusable factor
row usage’ni alohida chiqaradi. Toza 10% exploration screen ham mean accuracy’ni
`58.294%`ga tushirdi va virtual dead fractionni hal qilmadi, shuning uchun u
ham rad qilindi. Factor candidate pool’ni `8→32` qilish ham faqat `+0.065 pp`
plain factorizedga berdi, mustaqil controldan `−0.130 pp` qoldi va xarajatni
oshirdi. Oldingi exploration logidagi weight-shape bug natijasi dalil sifatida
ishlatilmaydi. Rank-32 pair basis ikki seedda `58.203%` mean accuracy va
`1.51172` mean CE berdi; plain factorizeddan ham yomon bo‘lgani uchun u ham
rad qilindi. Plain factorized smoke factor row’larning `86/87`tasini ishlatdi,
shuning uchun muammo factor row exposure emas, kombinatsiya representation’i
va final task-loss alignment tomonida. 3,000 qadamlik ikki-seed continuation’da
mustaqil 7,552 bank `68.620%` accuracy / `0.99017` CE, factorized bank esa
`66.888%` / `1.05428` berdi; factorized keyinroq yetib olmadi, farq `−1.732 pp`
gacha kengaydi. Shu sabab factorized virtual-capacity yo‘li sifat yechimi
sifatida yakuniy rad qilindi, faqat compression/runtime opt-in qoldi.

**Batafsil:** `results/P003_NATIVE_FACTORIZED_VIRTUAL_BANK_AUDIT_20260910.md`.

Global hierarchical router bilan factorized bank qayta tekshirilganda 300M
virtual bank ikki seedda `68.932%` accuracy / `0.97300` CE berdi va to‘g‘ridan-
to‘g‘ri independent 300M controldan `+0.638 pp` ustun chiqdi. Total params
`299.54M → 12.58M`, VRAM `5735 → 733 MB`, vaqt `458.65 → 142.74s` bo‘ldi.
Lekin 500M factorized-global `68.281%` bilan 300Mdan `−0.651 pp` pastladi;
scale monotonic emas. 300M variant `PROMISING OPT-IN`, lekin default yoki
yakuniy capacity yechimi emas; P-003 ochiq qoladi va 500M/700M/1B blind
expansion qilinmaydi.

**Scale audit:** `results/P003_NATIVE_FACTORIZED_GLOBAL_SCALE_AUDIT_20260910.md`.

300M virtual address count fixed holda factor row’larni `151→256` oshirish
ham sinab ko‘rildi: mean accuracy `59.232%→58.320%` tushdi, CE esa faqat
`1.41985→1.41663` yaxshilandi. Bu hard-quality gate’dan o‘tmaydi; 151 row’li
300M factorized-global variant saqlanadi, 256 row’li variant rad qilindi.

Address residual gipotezasi ham tekshirildi: 300M factorized-global bankka har
bir virtual address uchun rank-2 kichik residual MLP qo‘shildi. Ikki seed,
1,000 qadamlik screen’da mean accuracy `58.828%` bo‘lib baseline
`59.232%`dan `−0.404 pp` pastladi; mean CE `1.42390` bo‘lib baseline
`1.41985`dan yomonlashdi. Total params `12.58M→56.36M`, VRAM `733→1,413 MB`
va vaqt `48.06→67.83s` oshdi. Barcha `151/151` factor row ishlatilganiga
qaramay quality yaxshilanmadi. Shuning uchun per-address residual
**capacity/sifat yechimi sifatida rad qilindi**; opt-in API testlangan holda
saqlandi. Bu natija muammoni faqat yangi mustaqil address-parametrlar bilan
to‘ldirish emas, balki shared address-conditioned signal yoki assignment/
distillation kerakligini ko‘rsatadi.

**Address-residual audit:** `results/P003_NATIVE_FACTORIZED_ADDRESS_RESIDUAL_AUDIT_20260910.md`.

Global router bilan shared rank-4 pair-basis ham 300M bankda 3,000 qadamga
uzaytirib tekshirildi. Mean accuracy `68.125%` bo‘lib plain global baseline
`68.242%`dan `−0.117 pp` past; mean CE `0.99293` bilan `0.99648`dan biroz
yaxshi, lekin training vaqti `142.74→162.66s` oshdi. 1,000 qadam screen’da
hard accuracy farqi `−0.313 pp` edi. Shuning uchun pair-basis ham sifat yoki
sig‘im yechimi sifatida rad qilindi, opt-in component saqlandi. Keyingi sinov
factor jadvalidagi first/second slot simmetriyasini alohida reusable jadvallar
bilan yo‘qotadi.

**Global pair audit:** `results/P003_NATIVE_FACTORIZED_GLOBAL_PAIR_AUDIT_20260910.md`.

Factor bankdagi first/second slot simmetriyasini olib tashlash uchun ikki
alohida reusable jadval (`ordered_factor_slots=true`) sinab ko‘rildi. 300M,
3,000 qadam, ikki seedda mean accuracy `68.451%` bo‘lib shared-slot global
baseline `68.242%`dan `+0.208 pp` yuqori; mean CE `0.98753` bo‘lib
`0.99648`dan yaxshi chiqdi. Ikkala seed ham accuracy bo‘yicha baseline’dan
yuqori. Parametr `12.58M→14.49M`, vaqt `142.74→146.89s`, VRAM `733→754 MB`
oshdi; factor row exposure `151/151` bo‘lib qoldi. Bu hozirgi eng kuchli
reusable representation candidate, lekin virtual-address fragmentation hali
yuqori va 300→500 scale gate hali ochiq. Shuning uchun defaultga olinmadi,
500M ordered screen keyingi qadam qilindi.

**Ordered-slot audit:** `results/P003_NATIVE_FACTORIZED_ORDERED_SLOTS_AUDIT_20260910.md`.

500M ordered-slotning birinchi varianti `d_model=512` bilan ishga tushgani
aniqlandi, holbuki shared 500M baseline `d_model=384` edi; `69.818%` natija
shu sabab fair taqqoslash dalili emas va confounded deb belgilandi. Teng
`d_model=384` matched run’da ordered mean accuracy `68.073%`, CE `1.01038`
bo‘lib shared baseline `68.281% / 0.99648`dan yomon chiqdi. Shunday qilib
ordered slot 300Mda ijobiy, lekin 500M scaling muammosini mustaqil hal qilmaydi.
Bu P-003ni yopmaydi; keyingi yo‘l query-conditioned shared factor mixing.

Query-conditioned shared factor mixing (`scale=0.5`) ham 300Mda ikki seed,
1,000 qadam sinovdan o‘tkazildi. Mean accuracy `59.206%` bo‘lib ordered
baseline `59.036%`dan faqat `+0.169 pp`, shared-slot baseline `59.232%`dan esa
`−0.026 pp` qoldi; seedlar `−0.312/+0.651 pp` qarama-qarshi yo‘nalishda bo‘ldi.
Mean CE `1.41590` ordered baseline `1.41467`dan yomonlashdi. Shuning uchun
query mix consistency hard-quality fix sifatida rad qilindi; keyingi gipoteza
factor slot matritsalarining parametrsiz elementwise product interaction’idir.

**Query-mix audit:** `results/P003_NATIVE_FACTORIZED_QUERY_MIX_AUDIT_20260910.md`.

Parametrsiz elementwise product interaction (`scale=8`) ham 300M ordered
bankda tekshirildi. Mean accuracy `58.971%` bo‘lib ordered baseline `59.036%`
dan `−0.065 pp` past, mean CE `1.41736` bo‘lib yomonroq, VRAM esa
`754→952 MB` oshdi. Product interaction rad qilindi.

Factor1 state’ni o‘zgartirib, factor2 transformed state’da ishlaydigan serial
composition ham tekshirildi. 300Mda mean `68.359% / 0.99285 CE`, ordered
additive baseline `68.451% / 0.98753`dan past. Matched 500Mda mean
`68.138% / 1.00212 CE` bo‘lib shared baseline `68.281% / 0.99648`dan ham
past; VRAM `1,364 MB`. Serial composition scaling fix sifatida rad qilindi,
opt-in sifatida saqlandi. Bu natijalar local combination algebra’sidan ko‘ra
virtual-address routing/assignment fragmentation asosiy muammo bo‘lishi
mumkinligini kuchaytiradi.

**Product audit:** `results/P003_NATIVE_FACTORIZED_PRODUCT_AUDIT_20260910.md`.

**Serial audit:** `results/P003_NATIVE_FACTORIZED_SERIAL_AUDIT_20260910.md`.

Router key representation ham bank bilan moslashtirib tekshirildi: global
hierarchical router endi virtual address key’ini ikki reusable factor key’dan
hosil qila oladi. Unordered route-key-only control 500M, 3,000 qadamda mean
`68.138% / 0.99588 CE` bo‘lib oddiy global-key baseline
`68.281% / 0.99648`dan accuracy bo‘yicha `−0.143 pp` qoldi. Ordered bank +
shared route-key combined candidate esa `68.594% / 0.98665` berdi (`+0.313 pp`),
ammo bu ikki o‘zgarishning combined effect’i. U 300Mda `68.451%`, 500Mda
`68.594%`, 700M short screenda `58.815%` bo‘ldi; 700M virtual dead fraction
`63.44%`gacha oshdi. Shuning uchun route-key alone quality fix emas, combined
variant opt-in saqlanadi, 700M/1B blind expansion rad qilinadi. Asosiy ochiq
muammo virtual-address candidate assignment/fragmentation bo‘lib qolmoqda.

Bir xil 500M seed17 specialization diagnostikasida oddiy global key uchun
candidate/selected pair cosine `0.37746/0.38004`, dead circuit `46.05%` va
`4,603` unique selected circuit chiqdi. Ordered bank + shared factor-key uchun
bu `0.26475/0.27034`, `43.74%` va `4,742` bo‘ldi. Demak route-key alignment
fragmentatsiya va ortiqcha o‘xshashlikni kamaytiryapti, lekin hard accuracy
faqat `68.047% → 68.229%` (`+0.182 pp`) oshdi; capacity muammosi yechildi deb
bo‘lmaydi.

**Shared-route-key audit:** `results/P003_NATIVE_FACTORIZED_SHARED_ROUTEKEY_AUDIT_20260910.md`.

Factor-grid candidate pool (`4x8`) qisqa screenda `59.440%` mean accuracy
bergan bo‘lsa ham, 3,000 qadamda `68.047%`ga tushdi; ordered shared-route-key
baseline `68.594%` edi. Seed17 diagnostikasida candidate cosine `0.19770`gacha
pasaydi, ammo hard tanlangan subset cosine `0.32222`gacha oshdi. Shuning uchun
candidate xilma-xilligi oshgani bilan foydali subset tanlash muammosi hal
bo‘lmagan. Factor-pair interaction score (`scale=1`) ham 1,000 qadamda mean
`59.232%` bo‘lib grid va baseline’dan foydali ustunlik bermadi. Ikkalasi ham
quality fix sifatida rad qilindi, opt-in nazorat sifatida saqlandi.

**Candidate-grid audit:** `results/P003_NATIVE_FACTORIZED_CANDIDATE_GRID_AUDIT_20260910.md`.

Hidden factor-product interaction ham tekshirildi: 1,000 qadamda mean
`59.766%` bo‘lib `59.258%` baseline’dan kichik ijobiy signal berdi, ammo 3,000
qadamda `68.307%`ga tushdi, ordered shared-route-key baseline esa `68.594%`
bo‘ldi. VRAM `1,039 → 1,493 MB`, vaqt esa taxminan `30%` oshdi; selected
cosine `0.27389` bo‘lib route redundancy kamaymadi. Shuning uchun bu yo‘l ham
quality/scaling fix sifatida rad qilindi, faqat opt-in mathematical control
sifatida qoldi.

**Hidden-product audit:** `results/P003_NATIVE_FACTORIZED_HIDDEN_PRODUCT_AUDIT_20260910.md`.

Factor-level hidden gate ham tekshirildi. 1,000 qadamda ikki seed mean
`59.410%` bo‘lib kichik ijobiy signal berdi, lekin 3,000 qadamda `68.060%`
gacha tushdi; ordered shared-route-key baseline `68.594%` edi. Parametr faqat
`+6,304`, VRAM deyarli o‘zgarmadi, ammo sifat yaxshilanmadi. Shuning uchun
hidden gate scaling yechimi sifatida rad qilindi, opt-in control sifatida
saqlandi.

**Hidden-gate audit:** `results/P003_NATIVE_FACTORIZED_HIDDEN_GATE_AUDIT_20260910.md`.

10,000 qadamli teng-budget capacity auditida 300M ordered shared-route-key
mean `77.096% / 0.66621 CE`, 500M esa `77.292% / 0.65446 CE` berdi. Demak 500M
sig‘imi yetarli training budget bilan ishlayapti, lekin hard accuracy foydasi
faqat `+0.195 pp`. 500M 10k diagnostikada 300Mga nisbatan ko‘proq unique
address (`4,912 vs 3,926`), pastroq dead fraction (`39.00% vs 42.23%`) va
barcha factor rows ishlatilgan. Asosiy muammo “sig‘im umuman ishlamayapti” emas;
qo‘shimcha virtual kombinatsiyalar task-useful specializationga to‘liq
aylanmayapti. 700M/1B kengayishi yangi mexanizmsiz hozircha rad.

**Long-budget capacity audit:** `results/P003_NATIVE_FACTORIZED_LONG_BUDGET_AUDIT_20260910.md`.

500M ordered-slotning birinchi varianti `d_model=512` bilan ishga tushgani
aniqlandi, holbuki shared 500M baseline `d_model=384` edi; `69.818%` natija
shu sabab fair taqqoslash dalili emas va confounded deb belgilandi. Teng
`d_model=384` matched run’da ordered mean accuracy `68.073%`, CE `1.01038`
bo‘lib shared baseline `68.281% / 0.99648`dan yomon chiqdi. Shunday qilib
ordered slot 300Mda ijobiy, lekin 500M scaling muammosini mustaqil hal qilmaydi.
Bu P-003ni yopmaydi; keyingi yo‘l query-conditioned shared factor mixing.

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

### C-RUNTIME-VEC4-PACK-001 — Fixed-pack xotira ko‘chirishini vektorlashtirish

**Status:** `REJECTED AS MATERIAL OPTIMIZATION`
**Muammo:** runtime bottleneck; P-007 sifat muammosini yechmaydi
**Natija:** `float4` CUDA pack selected rows uchun exact parity berdi va ikki
seedli full-cascade graph timing o‘zgarishi B=1/B=8/B=32 da
`-0.19%/-0.05%/+0.06%` bo‘ldi. Eager mean `-0.91%/-1.25%/+0.14%`; bu graph
servingda material yutuq emas. Decode B=1 dagi pack-only penalty sabab adaptiv
fallback qo‘shildi. Variant opt-in benchmark sifatida saqlandi, default
o‘zgarmadi.
**Batafsil:** `results/RUNTIME_QWEN_VEC4_FIXED_PACK_AUDIT_20260909.md`.

### C-RUNTIME-DERIVED-POSITION-001 — Fixed-pack pozitsiyasini finalizerda hisoblash

**Status:** `REJECTED AS MATERIAL OPTIMIZATION`
**Muammo:** runtime bottleneck; P-007 sifat muammosini yechmaydi
**Natija:** Fixed-pack `packed_positions` bufferi olib tashlandi. Ikki seedli
full-cascade graph o‘zgarishi B=1/B=8/B=32 da `-0.86%/+0.03%/+0.05%`, eager
o‘zgarishi `-7.74%/-3.12%/-0.10%` bo‘ldi. Exact numerical parity va generation
saqlandi. Eager yutug‘i graph servingga ko‘chmagani uchun default o‘zgarmadi;
variant opt-in qoldi.
**Batafsil:** `results/RUNTIME_QWEN_DERIVED_POSITION_AUDIT_20260909.md`.

### C-P007-NATIVE-DELTA2-001 — Universal circuit correction amplitude 2x

**Status:** `REJECTED AS QUALITY FIX`
**Muammo:** P-007 / P-003
**Natija:** Router va circuit bankni o‘zgartirmasdan, `circuit_delta_scale=2.0`
ikki seedli 300M, 3000-qadamli screen’da tekshirildi. Shared route-key
baselinega nisbatan accuracy `68.451% → 68.620%` (`+0.169 pp`) bo‘ldi,
hard-task mean `33.366% → 33.398%` (`+0.033 pp`) xolos, CE esa
`0.99639 → 1.00105` yomonlashdi. Depth diagnostikasi 500Mda depth-2/3
correction normasi 300Mdan kuchliroq emasligini ham ko‘rsatdi. Universal
amplitude oshirish adoption uchun yetarli emas; opt-in control saqlandi.
**Batafsil:** `results/P007_NATIVE_DELTA2_DEPTH_AUDIT_20260910.md`.

### C-P007-NATIVE-TASK-CONTEXT-001 — Explicit task context for routing

**Status:** `REJECTED AS RELIABLE QUALITY FIX`
**Muammo:** P-007 / P-003 / P-005
**Natija:** 300M, ikki seed, 3000-qadamli screen’da router-only task context
`1.0×` umumiy accuracy’ni `+0.313 pp` oshirdi, lekin hard-task mean
`−0.163 pp` va dead traffic `+8.78 pp` yomonlashdi. `0.25×` variant ham
baseline’dan `−0.156 pp` accuracy va `−0.228 pp` hard-task mean qoldi.
Embeddingni state update’ga qo‘shish esa `−0.404 pp` berdi. API opt-in sifatida
saqlandi, default va route majburlash o‘zgarmadi.
**Batafsil:** `results/P007_NATIVE_TASK_CONTEXT_AUDIT_20260910.md`.

### C-P007-NATIVE-STEP-ADAPTER-001 — Step-specific circuit correction interface

**Status:** `PROMISING OPT-IN — VALIDATION OPEN`
**Muammo:** P-007 / P-003
**Natija:** Rank-8 step-specific low-rank residual adapter router va active
route budgetini o‘zgartirmadi. 300M 3k screen’da ikki seed mean accuracy
`+0.612 pp`, hard-task mean `+1.302 pp`, CE `−0.01671`; 10kda 300M
`+0.352 pp`, 500M `+0.026 pp` bo‘ldi. Hard-task mean ikkala scale’da
oshdi, lekin 500M overall scaling `−0.130 pp` va high-edge probe regressiyasi
saqlanib qoldi. Shuning uchun adapter default emas, hozirgi leading opt-in
candidate; keyingi ish high-edge/value representation validation.
**Batafsil:** `results/P007_NATIVE_STEP_ADAPTER_SCALE_AUDIT_20260910.md`.

### C-P003-NATIVE-EDGE-COVERAGE-001 — Low/high edge training coverage

**Status:** `PROMISING OPT-IN — 500M SCALING NEGATIVE`
**Muammo:** P-003 / P-007

**Natija:** 300M rank-8 step-adapterda 25% ikki-edge mix 10k, seed17/18
juftligida uniform accuracyni `+0.599 pp`, combination holdoutni `+0.629 pp`,
low-edge’ni `+17.340 pp` va high-edge’ni `+22.097 pp` yaxshiladi. Hard-task
mean ordinary probe’da `+2.268 pp` oshdi. 3k screen yomon ko‘ringani uchun
long-budget validation zarur bo‘ldi. Bu model body yoki active pathni
o‘zgartirmaydi; u faqat rare value regimesni trainingda ko‘paytiradi. Shu
recipe 500Mda qayta tekshirilganda edge foydasi saqlanib qoldi
(`+0.286/+0.920 pp` low/high), lekin uniform `−1.128 pp` va hard-task
`−2.192 pp` bo‘ldi. Shuning uchun recipe 300M uchun current reference opt-in,
500M esa scaling bo‘yicha salbiy nazorat; default va 700M/1B qarori ochiq
qoladi. Depth auditida 500M selected-pair cosine biroz kamaygan bo‘lsa ham,
depth-2/3 route-to-query ratio pasayib, uniform dead traffic `57.47% → 61.12%`
oshdi; ya’ni qo‘shimcha qatorlar active creditni kuchaytirmayapti.
**Batafsil:** `results/P003_NATIVE_EDGE_MIX_AUDIT_20260910.md`.
