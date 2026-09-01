# Neural Engine scale audit va keyingi eksperimentlar uchun taklif

Bu hujjat agent uchun amaliy handoff. Maqsad — hozirgi Neural Engine natijalarini 8B/13B/32B masshtabga olib chiqishdan oldin arxitekturaning yashirin yoriqlarini, training bottlenecklarini va falsifikatsiya qiluvchi testlarni aniq ajratish.

## 0. Qisqa hukm

Hozirgi asosiy g‘oya — **persistent recurrent state + hierarchical routing + ko‘p reusable low-rank micro-circuits + adaptive execution** — scale uchun tabiiy ko‘rinadi. Eng kuchli signal: total capacity 20M -> 100M oshganda active path ~2.04M atrofida qolmoqda, routing collapse kuzatilmagan va quality qulamagan.

Lekin hozirgi V0.12/V0.15 implementation’ni shunchaki 8B qilib yuborish tavsiya etilmaydi. Eng katta ochiq masala:

> **stored capacity oshishi hali yetarli darajada useful capability/intelligence oshishiga aylanmayapti.**

Shuningdek current PyTorch/AdamW trainer katta scale uchun haqiqiy sparse trainer emas. Inference arxitektura scale-friendly, current training stack esa 8B-friendly emas.

Quyidagi punktlarning barchasi qayta sinovdan o‘tkazilsin.

---

# 1. 2x sekin training nimani anglatadi

Agar dense 8B model uchun bir xil token budget 1 oy talab qilsa va Neural Engine aynan shu GPU sonida 2x sekin bo‘lsa, wall-clock ~2 oy bo‘ladi. Bu juda qimmat.

Lekin iqtisodiy jihatdan muhim metrika faqat wall-clock emas:

```text
Dense: 8 GPU x 1 oy = 8 GPU-month
NE:    1 GPU x 2 oy = 2 GPU-month
```

Bu holatda NE 2x sekin wall-clock bo‘lsa ham 4x kam GPU resource ishlatadi.

Shuning uchun RAM/GPU sparse training uchun maqsad:

- juda kam GPU bilan ishlash;
- wall-clock penaltyni imkon qadar <=2-3x ushlash;
- total GPU-hours va GPU-dollarni keskin tushirish.

Agar 1 GPU’da 8B pretraining 5-6 oyga cho‘zilsa, arzon bo‘lsa ham amaliy jihatdan yomon. Bunday joyda multi-GPU parallelism kerak bo‘ladi.

---

# 2. RAMdan circuitlarni ketma-ket chaqirish modelni avtomatik MoEga aylantirmaydi

Hozirgi computation flow:

```text
input
 -> state_t
 -> router(state_t)
 -> selected circuits
 -> state_t+1
 -> router(state_t+1)
 -> boshqa selected circuits
 -> state_t+2
 -> ...
 -> output
```

Keyingi route avvalgi circuitlar natijasidan yangilangan state asosida olinadi. Bu bitta global dynamic computational trajectory.

Shuning uchun:

```text
RAM -> GPU active circuit paging
```

faqat **memory placement** va execution backend masalasi. U architecture semantics’ni o‘zgartirmaydi.

### Qachon MoEga o‘xshab qolish xavfi bor?

Agar trainingda circuit bank segmentlari mustaqil o‘qitilsa:

```text
bugun circuit 1-1000
ertaga circuit 1001-2000
...
```

va ular bir xil end-to-end trajectory loss orqali bog‘lanmasa, independent experts paydo bo‘lishi mumkin.

To‘g‘ri sparse trainer bitta sample uchun to‘liq trajectory’ni saqlashi kerak:

```text
input
 -> state
 -> circuit 714
 -> state
 -> circuit 18291
 -> state
 -> circuit 42
 -> final loss
 -> gradient shu trajectory bo‘ylab orqaga
```

Faqat trajectoryga kirmagan 7.9B parametr update qilinmaydi.

**Talab:** training semantics end-to-end qolishi shart.

---

# 3. Eng katta arxitektura xavfi: capacity -> capability conversion juda sust

Hozirgi ikki-seed scaling signal:

- total capacity: ~20M -> ~100M (~5x)
- avg active path: ~2.04M -> ~2.04M
- active fraction keskin tushadi
- quality qulamaydi

Bu yaxshi.

Lekin mean quality improvement hozircha kichik:

```text
full mean:    ~+0.26 percentage point
held-out mean:~+0.52 percentage point
```

Bu shuni anglatadi:

> ko‘proq stored circuits bor, lekin ular hali ko‘proq useful knowledge/skillga samarali aylanmayapti.

Agar bu yechilmasa:

```text
8B total = kichik useful core + juda katta dormant bank
```

bo‘lib qolishi mumkin.

### Shu sabab 8B’dan oldingi asosiy milestone

100M -> 300M -> 500M’da benchmarkni boyitib, **quality aniq oshishi** ko‘rsatilishi kerak.

Faqat active fractionning tushishi yetarli emas.

---

# 4. Hierarchical router scale uchun juda kuchli tomon

Current router barcha circuitsni score qilmaydi. Tree traversal + local candidate pool ishlaydi.

Taxminiy routing complexity:

```text
O(depth * branch + candidate_pool)
```

bank size N oshganda depth taxminan log(N) bilan o‘sadi.

Bu inference uchun juda yaxshi property.

Masalan yuz minglab circuitga ham branch=8 bilan nisbatan kichik depth yetadi.

**Bu core design saqlansin.** Dense all-circuit scoringga qaytilmasin.

---

# 5. Lekin current coverage loss katta scale’da portlashi mumkin

Current soft coverage mexanizmi tree bo‘yicha soft path probabilitylarni yoyadi va leaf candidate windowsga mass tarqatadi.

Depth oshganda leaf count eksponential:

```text
8^5 = 32,768
8^7 ~ 2.1 million
```

8B masshtabda bu training-only auxiliary loss juda qimmatlashishi mumkin.

### Alternative lar sinansin

- sampled load-balancing loss;
- EMA circuit utilization;
- stochastic route exploration;
- local subtree balancing;
- random route perturbation;
- Gumbel/straight-through exploration;
- per-leaf reservoir statistics;
- low-cost entropy/load regularizer.

**Acceptance:** router coverage/control cost bank size bilan deyarli log/sublinear o‘sishi kerak.

---

# 6. Current AdamW trainer katta scale uchun real sparse trainer emas

Hozir optimizer:

```python
AdamW(model.parameters())
```

va circuit bank katta dense Parameter tensorlar sifatida saqlanadi.

Forward selected IDs bo‘yicha sparse bo‘lsa ham training infrastructure:

- full parameter storage GPUga bog‘langan;
- optimizer state bank size bilan o‘sadi;
- 8B’da VRAM va optimizer state bottleneck bo‘ladi.

### Muhim xulosa

**Inference architecture scale-friendly. Current trainer scale-friendly emas.**

Yangi trainer ixtiyoriy optimizatsiya emas; 8B+ uchun deyarli majburiy.

---

# 7. “O‘zimiz yasagan multimetr” muammosini qanday yo‘qotamiz

Architecture va trainer’ni birdan almashtirib, natija yomon chiqsa sababni ajratib bo‘lmaydi.

Shuning uchun yangi sparse trainer kichik scale’da current AdamW bilan qat’iy A/B qilinsin.

### Test protocol

20M va 100M’da:

```text
A = current PyTorch AdamW
B = new sparse/offloaded trainer
```

Iloji boricha:

- bir xil initialization;
- bir xil seed;
- bir xil minibatch stream;
- bir xil LR schedule;
- bir xil loss;
- bir xil route decisions (diagnostic mode’da mumkin bo‘lsa replay);
- bir xil gradient clipping.

Compare:

- step-by-step train loss;
- validation accuracy;
- held-out accuracy;
- route utilization;
- circuit parameter deltas;
- convergence speed;
- optimizer numerical drift.

### Go criterion

Sparse trainer AdamW referencega yaqin convergence va final quality bersa, keyin 500M/1B+ faqat sparse trainer bilan davom etish mumkin.

---

# 8. Avval “new optimizer” emas, sparse/lazy Adam semantics sinansin

AdamW’dan birdan matematik jihatdan butunlay voz kechish shart emas.

Avval quyidagi system qurilishi tavsiya etiladi:

```text
CPU RAM:
  dormant weights
  dormant optimizer states (m/v)
  circuit metadata

GPU:
  shared controller
  router
  active circuit cache
  active gradients
```

Har step:

```text
route
 -> selected circuit IDs
 -> prefetch/cache active circuits
 -> forward
 -> backward
 -> faqat active circuit gradients
 -> faqat active circuit m/v update
 -> dirty circuits RAMga qaytadi
```

Bu **lazy/sparse AdamW** bo‘lishi mumkin.

Keyin boshqa optimizerlar alohida ablation qilinsin.

---

# 9. Eng katta systems bottleneck: PCIe bandwidth va locality

Agar har stepda random katta active set RAMdan GPUga tashilsa, GPU kutib qoladi.

### Shuning uchun trainer/router birga quyidagilarni optimallashtirishi kerak

- circuit cache;
- route locality;
- batch ichida route grouping;
- next-step prefetch;
- asynchronous H2D transfer;
- pinned memory;
- cache hit-rate instrumentation;
- hot circuit tier GPU VRAM’da;
- warm circuit tier host RAM’da;
- cold circuit tier optional NVMe’da;
- route-aware mini-batch scheduling.

### O‘lchanadigan metrikalar

- H2D bytes/sample;
- H2D GB/s;
- GPU idle fraction;
- cache hit-rate;
- unique circuits/batch;
- repeated circuits/batch;
- step latency breakdown;
- optimizer bytes touched;
- total RAM bandwidth;
- wall-clock vs AdamW baseline.

---

# 10. 384-dimensional recurrent state kelajakdagi information bottleneck bo‘lishi mumkin

Hozir deyarli barcha reasoning 384-d state orqali o‘tadi.

Bank 20M, 100M yoki 8B bo‘lsa ham working state bir xil bo‘lsa, ulkan knowledge bankdan foydalanish qobiliyati shu kichik state bilan cheklanib qolishi mumkin.

Analogiya:

> ulkan kutubxona, lekin juda kichik ish stoli.

### Sinovlar

State dimension sweep:

- 256
- 384
- 512
- 768
- 1024

Lekin faqat bitta kattaroq dense state yechim emas. Quyidagilar ham sinansin:

### Multi-slot working memory

```text
state = [slot_1, slot_2, ..., slot_M]
```

Masalan:

- 4/8/16/32 slots;
- har circuit barcha slotga emas, selected slotga yozadi;
- router global summary + local slot context oladi;
- slot write/read structured bo‘ladi.

Maqsad: working memory capacity oshsin, lekin active compute dense kvadrat o‘smasin.

---

# 11. Input reinjection bo‘yicha hozirgi signal

Input signalni har recurrent stepda kamaytirish sinovi reference’ni yenga olmadi.

Half reinjection depth-3ni ozgina yaxshilagan bo‘lsa ham overall/held-out quality pasaydi.

Demak oddiy:

```text
input reinjectionni kamaytirish
```

memory bottleneckni hal qilmaydi.

Keyingi memory ishlari explicit structured memory/write mexanizmlariga qaratilsin.

---

# 12. Gated memory write bo‘yicha signal

Generic learned write gate:

- held-outda ba’zi foyda berdi;
- full accuracy pasaydi;
- active fraction oshdi;
- throughput pasaydi.

Bu shuni ko‘rsatadi:

> state preservation/memory muhim bo‘lishi mumkin, lekin generic dense gate juda qimmat va yetarlicha yaxshi emas.

### Keyingi variantlar

- slot-level write gate;
- sparse write addresses;
- low-rank write controller;
- per-slot confidence;
- append/overwrite hybrid memory;
- differentiable stack/queue-like scratch memory;
- cheap structured recurrent memory.

Har memory mexanizmi composition benchmarkda tekshirilsin.

---

# 13. Micro-circuitning o‘zi hozir juda sodda

Current circuit taxminan:

```text
state_dim -> rank 16 -> state_dim
```

low-rank transform.

Bu juda arzon, lekin murakkab computation uchun yetarli bo‘lmasligi mumkin.

### Circuit family ablations

Bir xil active MAC/param budget ostida:

1. rank 8 / 16 / 32 / 64;
2. gated low-rank circuit;
3. 2-layer tiny MLP circuit;
4. multiplicative circuit;
5. residual operator circuit;
6. per-circuit normalization/no normalization;
7. state-slot-specific circuit;
8. shared basis + per-circuit coefficients.

Maqsad: capacity oshganda qo‘shimcha circuits haqiqiy yangi skillga aylana oladimi?

---

# 14. Parallel mix vs haqiqiy composition

Current defaultda selected circuits parallel outputlari weighted sum qilinadi.

Oldingi serial circuit ablation 2x sekinlashib, depth-3ni yaxshilamadi. Bu “serial composition yomon” degani emas; current serial implementation yomon tradeoff berdi.

### Alternative composition graphs

- 2-stage grouped composition;
- pairwise circuit chains;
- tree-of-circuits;
- one router stepda 2 substeps;
- cheap affine circuit -> nonlinear circuit;
- recurrent route between circuit groups;
- learned operator DAG with hard budget.

Har bir variant active MAC va wall-clock bilan birga baholansin.

---

# 15. Hard routing exploration / credit assignment katta scale’da xavf

Current tree path hard argmax bilan tanlanadi.

Katta bankda muammo:

> boshida noto‘g‘ri subtree tanlansa, task gradient boshqa yuz minglab circuitsga yetmaydi.

Bu capacity utilizationning sust scalingiga sabab bo‘lishi mumkin.

### Sinovlar

- epsilon-random subtree exploration;
- top-2/top-m branches faqat trainingda;
- stochastic categorical path;
- Gumbel-softmax/straight-through;
- occasional route mutation;
- novelty bonus for underused circuits;
- parent-child split inheritance;
- route replay buffer;
- alternative-route contrastive loss.

Inference esa hard/sparse bo‘lib qolishi mumkin.

---

# 16. “Grow the model” strategiyasi: 8B’ni randomdan yaratish shart emas

Bu architecture uchun incremental growth juda tabiiy.

### Tavsiya etilgan growth algorithm

```text
100M trained
 -> hot / overloaded / multi-context circuits aniqlanadi
 -> selected circuits split/clone qilinadi
 -> child circuits parent weightsdan boshlaydi
 -> router subtree kengayadi
 -> specialization training
 -> 200M
 -> 500M
 -> 1B
 -> ...
```

Masalan:

```text
C -> C1, C2
```

C1/C2 random emas, C’dan initialize qilinadi.

### Bu uch muammoni birga yechishi mumkin

1. yangi capacity boshidan useless dormant bo‘lmaydi;
2. router yangi capacity qayerda ekanini biladi;
3. existing knowledge continuity saqlanadi.

### Split trigger lar

- circuit utilization yuqori;
- gradient conflict yuqori;
- diverse context clusters;
- high loss samples ko‘p route qilinadi;
- circuit embedding multimodal bo‘lib qoladi.

### Merge/prune ham sinansin

- redundant circuits merge;
- dead circuits recycle;
- low-value circuits reset/split from useful parent.

---

# 17. 8B scale’da approximate circuit count va routing depthni oldindan simulyatsiya qilish

Current state_dim/rank/circuit footprintdan kelib chiqib 8B’da yuz minglab circuits chiqishi mumkin.

8B’ni actual train qilishdan oldin dummy bank bilan:

- 100k circuits;
- 250k;
- 500k;
- 1M;

router-only va paging-only stress test qilinsin.

Actual weights random bo‘lishi mumkin; maqsad systems scale:

- routing latency;
- candidate lookup;
- cache indexing;
- RAM footprint;
- coverage/control overhead;
- batch route diversity.

Bu test architecture trainingdan juda arzon.

---

# 18. LLM uchun current input frontend tayyor emas

Current structured slot encoder synthetic tasks uchun kuchli, lekin long natural-language context uchun yetarli emas.

8B language model qilish uchun alohida sequence frontend kerak.

### Attention ishlatish majburiy emas

Variantlar:

- streaming recurrent encoder;
- hierarchical chunk compressor;
- state-space frontend;
- local convolution + recurrent memory;
- chunk -> memory slots;
- multi-timescale recurrent state;
- token stream -> routed state updates.

### Muhim qoida

Circuit/router core bilan sequence frontend alohida modullar sifatida test qilinsin.

Architecture core synthetic/reasoning tasklarda isbotlanmaguncha full LLM frontend bilan chalkashtirilmasin.

---

# 19. Benchmark fairness: Transformerga ham teng input inductive bias berish kerak

Numeric/Fourier value encoding Neural Enginega synthetic numeric tasksda foyda beryapti.

Transformer baseline esa oddiy embedding oladi.

Shu sabab 71.98% vs 51.02% farqning hammasini architecture superiority deb talqin qilish mumkin emas.

### Rigorous baselines

Kamida:

1. Transformer + same numeric/Fourier input representation;
2. Transformer standard;
3. GRU/LSTM baseline;
4. MLP baseline;
5. state-space/Mamba-like small baseline (agar implementation oson bo‘lsa);
6. simple MoE baseline;
7. Neural Engine.

Bir xil:

- total params;
- training examples;
- optimizer;
- steps;
- input information;
- seed set;
- evaluation split.

### Eng muhim report

Quality vs:

- total params;
- active params;
- MACs;
- parameter bytes read;
- wall-clock;
- VRAM;
- GPU-hours.

---

# 20. Adaptive halting bo‘yicha ehtiyotkor talqin

Current adaptive halting task depth labelidan supervision oladi.

Shuning uchun:

```text
depth-1 -> 1 step
depth-2 -> 2 step
depth-3 -> ~3 step
```

natijasi mechanism ishlashini ko‘rsatadi, lekin general self-discovered difficulty emas.

Real LLM’da true depth label bo‘lmaydi.

### Keyingi adaptive compute testlari

- confidence-based halt;
- expected future loss reduction;
- ponder cost;
- entropy threshold;
- self-consistency trigger;
- learned value-of-computation head;
- unsupervised halting with compute penalty.

Task depth label faqat oracle/reference sifatida qolishi mumkin.

---

# 21. Benchmarkni ancha qattiqlashtirish kerak

Current 15-task synthetic suite architecture smoke test sifatida yaxshi, lekin scale claim uchun yetarli emas.

### Yangi benchmark oilalari

#### Arithmetic / symbolic
- longer chains;
- unseen operator compositions;
- modular arithmetic;
- variable-length expressions;
- nested conditionals;
- algebraic rewrites.

#### Algorithmic
- sorting;
- permutation;
- stack/queue tasks;
- copying/reversal;
- pointer chasing;
- associative recall.

#### Graph
- BFS-like reachability;
- shortest path small graphs;
- multi-hop relations;
- graph transformations.

#### State machines
- hidden state transitions;
- delayed dependencies;
- variable execution depth.

#### Compositional generalization
- train primitives separately;
- test unseen compositions;
- train A+B, B+C, test A+C;
- hold out operator orders;
- hold out chain lengths.

#### Distribution shift
- unseen ranges;
- unseen lengths;
- unseen graph sizes;
- unseen combinations.

### Goal

Dormant capacity yangi task families va new compositionsni storage qilib, active path kam qolgan holda quality oshirishi kerak.

---

# 22. Capacity scaling experiment dizayni

Model sizes:

```text
20M
50M
100M
200M
300M
500M
```

8B hozircha shart emas.

Har size uchun kamida 3 seed, ideal 5 seed.

### Ikki alohida scaling mode

#### A. Fixed-active scaling

Active path deyarli bir xil qoladi.

Savol:

> more stored capacity qualityni oshiradimi?

#### B. Mild-active scaling

Active compute sekin oshadi, masalan:

```text
20M  -> 2M active
100M -> 3M
500M -> 5M
```

Savol:

> kichik active growth katta quality growth bera oladimi?

Bu real katta model uchun fixed 2Mdan ko‘ra tabiiyroq bo‘lishi mumkin.

---

# 23. 8B’dan oldingi GO / NO-GO mezonlari

## GO uchun tavsiya

Quyidagilarning katta qismi bajarilsin:

1. 100M -> 300/500M’da quality statistik aniq oshadi;
2. active fraction keskin tushishda davom etadi;
3. held-out composition quality ham oshadi;
4. router collapse yo‘q;
5. dead/unused capacity nazoratda;
6. sparse trainer AdamW referencega yaqin quality beradi;
7. sparse trainer GPU-hours bo‘yicha sezilarli yutuq beradi;
8. RAM paging GPUni haddan tashqari idle qoldirmaydi;
9. routing/control overhead sublinear qoladi;
10. long-context frontend alohida smoke testdan o‘tadi;
11. fair Transformer+same-input baseline ustidan compute/quality Pareto yutug‘i saqlanadi.

## NO-GO / rethink signal

- 500M quality 20/100Mdan oshmaydi;
- useful circuit utilization kamayadi;
- router exploration bank kattalashganda buziladi;
- sparse trainer convergence sezilarli yomonlashadi;
- RAM paging >50% wall-clockni yeb qo‘yadi;
- active compute bank size bilan deyarli linear o‘sishga majbur bo‘ladi;
- richer benchmarkda architecture advantage yo‘qoladi.

NO-GO “projectni to‘xtatish” degani emas; bottleneckni topib core designni qayta ko‘rish degani.

---

# 24. 8B/13B training uchun ideal target system

Agar yuqoridagi testlar yaxshi chiqsa, target:

```text
CPU RAM:
  8B/13B full bank
  optimizer states
  circuit metadata

GPU VRAM:
  shared controller
  active working set
  gradient buffers
  hot cache
```

### Ideal 8B target misol

Bu hozircha hypothesis, benchmark bilan isbotlanishi kerak:

```text
Total: 8B
Average training-active: 50M-300M range
RAM: 128-256GB+
GPU VRAM: 12-24GB+
```

Agar dense 8Bga nisbatan GPU-hours keskin kamayib, wall-clock <=2-3x penaltyda qolsa — bu alohida katta systems result.

---

# 25. Multi-GPU architecture uchun tabiiy partition

Neural Engine bank-based bo‘lgani uchun expert-parallel MoEga o‘xshash implementation ishlatish mumkin, lekin semantics reusable micro-circuit trajectory bo‘lib qoladi.

Variantlar:

- bank shards per GPU;
- hot circuits replicated;
- router locality penalty;
- batch grouped by destination GPU;
- asynchronous remote-circuit fetch;
- shared recurrent state transfer.

Muhim test:

> communication active working set bilan scale qiladimi yoki total bank bilan?

Agar active set bilan scale qilsa, architecture katta cluster uchun ham qulay.

---

# 26. Training costni total params emas active params bilan scale qilish — alohida asosiy research goal

Projectda endi uchta mustaqil hypothesis bor:

## A. Sparse inference ishlaydimi?

Hozircha kuchli positive signal.

## B. More total capacity -> more capability bo‘ladimi?

Hozircha promising, lekin weak.

## C. Training cost ham active compute bilan scale qiladimi?

Hali isbotlanmagan.

Agar B va C ham “ha” bo‘lsa, projectning qiymati faqat tez inference emas.

U:

> katta modelni arzon hardware’da train qilish paradigmasi

bo‘lishi mumkin.

---

# 27. Agent uchun tavsiya etilgan aniq ish tartibi

## Phase 1 — rigor / fairness

1. Transformerga same numeric encoder ber.
2. GRU/MLP/simple MoE baseline qo‘sh.
3. 3-5 seed benchmark qil.
4. richer composition benchmark yarat.
5. current V0.12/V0.15ni qayta benchmark qil.

## Phase 2 — capacity utilization

6. 200M/300M model qo‘sh.
7. fixed-active va mild-active scaling qil.
8. hard-routing exploration ablations qil.
9. circuit family/rank ablations qil.
10. growth/split strategy prototype qil.

## Phase 3 — memory/reasoning

11. state_dim sweep.
12. multi-slot state prototype.
13. cheap structured write/read memory.
14. alternative circuit composition graph.
15. unsupervised adaptive halting.

## Phase 4 — sparse trainer

16. current AdamW reference checkpoint/curve freeze qil.
17. active-only lazy AdamW prototype.
18. 20M exact A/B.
19. 100M A/B.
20. CPU RAM offload.
21. GPU circuit cache + async prefetch.
22. measure GPU idle/cache hit/H2D bytes.

## Phase 5 — systems scale without expensive training

23. dummy 100k/250k/500k/1M circuit bank routing+paging stress test.
24. 500M real sparse training.
25. 1B short run.
26. faqat shundan keyin 8B short architecture validation.

---

# 28. Har yangi experiment uchun report format

Har run natijasida quyidagilar yozilsin:

```text
commit_sha
config
seed
hardware
model_total_params
unique_active_params
avg_active_params
active_fraction
active_circuits
avg_steps
train_steps
train_tokens/examples
train_seconds
GPU-hours
peak_vram
host_ram_peak
H2D_bytes_per_sample
cache_hit_rate
GPU_idle_fraction
train_loss_curve
validation_accuracy
heldout_accuracy
composition_accuracy
per-depth/per-task accuracy
router entropy
circuits used
dead circuit fraction
max route load
throughput
latency
analytical MACs
parameter-read proxy
```

Har report oxirida:

```text
FAIL / WEAK SIGNAL / STRONG SIGNAL
```

va keyingi falsifikatsiya testini yozish kerak.

---

# 29. Hozirgi architecture haqida yakuniy baho

## Juda kuchli tomonlar

- hierarchical routing total bankni dense scan qilmaydi;
- active compute stored capacitydan ajralishi real ko‘rsatildi;
- 20M -> 100M’da active path deyarli constant;
- routing collapse yo‘q;
- route outputga causal ta’sir qilishi ko‘rsatilgan;
- micro-circuit tensors structured/contiguous;
- recurrent state circuitlarni global trajectoryga ulaydi;
- adaptive execution mechanism mavjud;
- real CUDA throughput advantage mavjud.

## Asosiy ochiq xavflar

- stored capacity -> useful capability conversion;
- small recurrent state bottleneck;
- hard routing exploration/credit assignment;
- scalable training/load balancing;
- RAM/GPU paging throughput;
- circuit composition;
- long-context frontend;
- benchmark fairness;
- supervised haltingdan general self-haltingga o‘tish.

## Final recommendation

**Hozirgi V0.12/V0.15ni shunchaki 8B qilib train qilmang.**

Avval 300M/500Mda:

1. quality scalingni isbotlang;
2. richer composition benchmarkda yutug‘ini saqlang;
3. sparse trainerni AdamW referencega qarshi validate qiling;
4. RAM/GPU pagingni real throughput bilan o‘lchang.

Agar shular o‘tsa, 8B experiment ko‘r-ko‘rona qimor emas, kontrolli engineering scaling testga aylanadi.

Eng katta potential yutuq:

> **model capacity juda katta bo‘lishi mumkin, lekin inference ham training ham faqat kichik, task-dependent working set bilan ishlaydi.**

Agar bu 500M -> 1B -> 8B’da saqlansa, Neural Engine’ning qiymati oddiy sparse inference’dan ancha katta bo‘ladi.
