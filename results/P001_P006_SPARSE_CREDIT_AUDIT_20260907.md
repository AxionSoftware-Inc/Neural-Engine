# P-001–P-006: nazoratli audit va minimal sparse-credit patch

Sana: 2026-09-07. **Credit patch: adoption uchun RAD ETILDI.** Instrumentation va
qayta ishlatiladigan benchmark qoldirildi. Mavjud default, router arxitekturasi,
circuit/recurrent body implementatsiyasi va checkpoint fayllari o'zgartirilmadi.

Bu hisobot olti fundamental muammo yechilganini da'vo qilmaydi. P-003 uchun
matched-training sababiy tajribasi hali bajarilmagan; uning o'rniga mavjud
checkpointlarning cheklovlari ochiq ko'rsatilgan matched-evaluation bajarildi.

## Nimalar qo'shildi

- `neural_engine/sparse_audit.py`: circuit ledger, final-class margin objective,
  faqat sampled circuit rowlarga qo'shimcha gradient, yangi cost-accounting.
- `benchmark_sparse_credit_audit.py`: seed17/18 to'rtta bank-credit controli,
  frozen-bank static/on-policy calibration, exact one-decision oracle,
  functional ablation, cascade diagnostikasi va 20M/50M/100M read-only audit.
- `tests/test_sparse_audit.py`: formula, gradient isolation, replay, optimizer
  update va backward-compatible JSON testlari.

Eski instrumentation funksiyalari/JSON kalitlari o'zgarmagan. Yangi ma'lumot
`sparse_audit_v2` va `training_cost_v2` bloklarida. Inference latency o'lchashda
ledger yoki exhaustive oracle ishlamaydi.

## Protokol

- Native 32-bank, K=2, M=8, T=3, seeds 17/18 mavjud frozen checkpointlaridan.
- CPU, 2 thread. GPU training/inference job boshlanmadi.
- Bank-credit: 500 update, fresh 60 misol/batch, AdamW LR=1e-4, weight decay=0,
  gradient clipping=1. Router, encoder, GRU va output-head vaznlari muzlatilgan.
  Har update'da 15 misol uchun bitta recurrent qarorda bitta circuit almashtirildi.
- Inference har qarorda ikkita circuit ishlatadi. Training replay ham sparse
  trajectory; barcha 32 body bir vaqtning o'zida hisoblanmaydi.
- Main loss CE. Extra row-credit final output CE yoki multiclass hinge:
  `relu(1 + max_wrong_logit - target_logit)`. Bu oddiy softplus/logsumexp orqali
  CE'ni boshqa nom bilan qayta yozish emas.
- Extra branch gradienti faqat sampled replacement rowlarga 0.1 koeffitsiyent
  bilan qo'shiladi. Router/controller auxiliary gradient olmaydi.
- Control ham ayni probe forward/backward budgetni sarflaydi, lekin extra
  gradientni tashlab yuboradi. Load-balancing loss qo'shilmagan.
- Held-out quality: 1,920 misol/seed. Exact audit: boshqa RNG'dan 30 misol,
  uch qarorda barcha 496 juftlik, jami 90 oracle decision/seed/arm.
- Exact regret har armning o'z bank/state/suffixida o'lchandi. Shuning uchun
  gapning qisqarishi yolg'iz o'zi improvement emas: CE va hard accuracy ham gate.
  Bu GLOBAL on-policy trajectory oracle emas.

Arm ma'nolari:

- frozen: asl checkpoint, qo'shimcha o'qitishsiz;
- control: main CE bilan faqat bank o'qitiladi;
- uniform_margin: uniform sampled row + final margin credit;
- underused_ce: kam gradient olgan rowlarga ustunlik + final CE credit;
- underused_margin: kam gradient olgan rowlarga ustunlik + final margin credit;
- calibration_static/on_policy: alohida 200-update tajriba, BANK MUZLATILGAN,
  faqat mavjud router keylari o'qitiladi. Ikkalasida ayni sakkiz calibration
  input batch, schedule, optimizer va replay budget. Static cached query/targetni,
  on_policy esa hozirgi model bilan qayta hisoblangan final-margin targetni oladi.
  Bu to'liq DAgger replay-mixture tadqiqoti emas, fresh re-labelling controli.

## P-001 va P-005: birlashtirilgan sifat jadvali

CE kichik, accuracy/recall katta bo'lgani yaxshi. Regret CE birliklarida.
Recall: candidate ichida kamida bitta exact CE-optimal juftlik to'liq mavjudligi,
1e-6 tie tolerance bilan. Jadvaldagi candidate pool doimo M=8.

| Seed | Arm | CE | Hard acc | Recall | Retrieval regret | Selection regret | Mean regret | p95 regret |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 17 | frozen | 2.34333 | 49.583% | 5.556% | 0.15412 | 0.26639 | 0.42051 | 1.92123 |
| 17 | control | 2.30807 | 50.208% | 8.889% | 0.14809 | 0.21694 | 0.36503 | 1.42176 |
| 17 | uniform_margin | 2.30808 | 50.313% | 8.889% | 0.14683 | 0.21536 | 0.36219 | 1.41555 |
| 17 | underused_ce | 2.30815 | 50.313% | 8.889% | 0.14818 | 0.21643 | 0.36460 | 1.41700 |
| 17 | underused_margin | 2.30764 | 50.313% | 8.889% | 0.14666 | 0.21523 | 0.36189 | 1.41225 |
| 17 | calibration_static | 2.34358 | 49.271% | 5.556% | 0.16154 | 0.23361 | 0.39515 | 1.61882 |
| 17 | calibration_on_policy | 2.34573 | 49.271% | 5.556% | 0.16101 | 0.24423 | 0.40524 | 1.61931 |
| 18 | frozen | 2.57319 | 45.677% | 6.667% | 0.18683 | 0.28181 | 0.46864 | 1.70589 |
| 18 | control | 2.57459 | 46.250% | 7.778% | 0.17662 | 0.34363 | 0.52025 | 2.28830 |
| 18 | uniform_margin | 2.57452 | 46.354% | 7.778% | 0.17275 | 0.34102 | 0.51377 | 2.27238 |
| 18 | underused_ce | 2.57541 | 46.250% | 7.778% | 0.17310 | 0.34321 | 0.51631 | 2.27979 |
| 18 | underused_margin | 2.57388 | 46.302% | 7.778% | 0.17289 | 0.34117 | 0.51406 | 2.27485 |
| 18 | calibration_static | 2.56661 | 45.938% | 6.667% | 0.19994 | 0.27536 | 0.47529 | 1.72231 |
| 18 | calibration_on_policy | 2.56983 | 45.990% | 6.667% | 0.20710 | 0.36193 | 0.56902 | 1.91175 |

Oldingi P001 retrieval-window hisobotida M oshishi retrieval regretni kamaytirib,
selection regretni oshirgani tasdiqlangan. Bu run uni M=16/24/32 "yechimi"ga
aylantirmadi; dense diagnostic inference yo'liga ko'chirilmadi. Hozirgi kichik
oracle sample oldingi 510 misollik recall foizlari bilan aynan teng bo'lishi
kutilmaydi. Mavjud dalillar: `P001_RETRIEVAL_WINDOW_S17_S18.md`,
`P001_SPARSE_OUTPUT_SIGNATURE_S17_S18_LOCAL.md`.

## P-002: gradient ko'paydi, sifat deyarli o'zgarmadi

| Seed | Arm | Eng kam nonzero-gradient step/500 | Eng kam actual update/500 | Greedy dead/32 |
|---:|---|---:|---:|---:|
| 17 | control | 2 | 129 | 1 |
| 17 | uniform_margin | 109 | 499 | 1 |
| 17 | underused_ce | 232 | 499 | 1 |
| 17 | underused_margin | 144 | 499 | 1 |
| 18 | control | 290 | 498 | 0 |
| 18 | uniform_margin | 344 | 500 | 0 |
| 18 | underused_ce | 387 | 500 | 0 |
| 18 | underused_margin | 346 | 500 | 0 |

Seed17'da eng kam o'qigan rowning gradient exposure'i 2 dan 144 qadamga chiqdi,
lekin mean accuracy foydasi juda kichik. Demak shu 32-bankda "ko'proq gradient
yetkazishning o'zi" asosiy sifat bottleneckini yechmadi. Bu katta banklar haqida
universal inkor emas.

Muhim farq: nonzero-gradient soni actual-update soni bilan teng emas. Adam
momentum sabab oldin gradient olgan row keyingi zero-gradient qadamlarda ham
o'zgaradi. Patch sparse credit beradi, lekin **sparse optimizer yoki end-to-end
sparse training** da'vo qilmaydi.

JSON har circuit uchun main usage, phase bo'yicha forward count, gradient norm,
nonzero-gradient step, haqiqiy update count, dead/under-trained flag, task usage
matritsasi va entropy beradi. Under-trained diagnostik threshold: kuzatilgan
qadamlarning 5 foizidan kam nonzero-gradient; bu universal ilmiy threshold emas.

Functional control: bir routed circuit hissasi zero-weight bilan olib tashlanadi;
qolgan butun route, weight/gain saqlanadi va final CE farqi task bo'yicha yoziladi.
Task-conditional ablation counts ham yozilgan: kuzatilmagan cell nol ta'sir
degani emas. Kamida ikki kuzatuv va >0.01 CE foyda bo'lgan circuit/task celllar
control -> underused_margin: seed17 **13 -> 13**, seed18 **9 -> 9**.
Yangi foydali specialization isbotlanmadi. Usage entropy yoki weight farqini
"circuitlar turli foydali algoritmlar o'rgandi" deb talqin qilish mumkin emas.

## P-004: prefix va suffix alohida

Barcha tekshirilgan changed-step prefix query error = **0**. Natural suffix
va fixed suffix ta'siri alohida o'lchandi. Masalan underused_margin armida:

- seed17, step1 (zero-based): natural-suffix gain -0.00474 CE, fixed-suffix
  gain -0.04594; rerouting CE ta'siri -0.04120, ya'ni suffix qisman yordam beradi;
- seed18, step1: natural -0.11897, fixed -0.05866; rerouting ta'siri +0.06031,
  ya'ni suffix zarar qo'shadi;
- oxirgi stepda suffix yo'q va farq 0.

Bir xil input va bir xil alternativ ID juftliklari qayta baholanganda eski
calibration targetlarining mean absolute drifti 0.04664 / 0.10575 CE bo'ldi.
Oldin aniq (|gain|>0.02) targetlarning sign agreement'i 93.88% / 91.07%.
Bu drift borligini ko'rsatadi, data aggregation albatta foydali ekanini emas.

Fresh on-policy calibration static controldan ikki seedda ham CE bo'yicha
yomonroq chiqdi; seed18 mean/p95 regret ham sezilarli yomonlashdi. Shuning uchun
shu 200-update final-margin refit recipe'si qabul qilinmaydi. Oldingi 5000-step
P004 tajribasining rad qarorini bu kichik run bekor qilmaydi. To'liq DAgger yoki
boshqa aggregation nisbatlari universal rad etilgani haqida xulosa yo'q.

## P-003: qayta baholash va ochiq qolgan sabablar

Mavjud v12 checkpointlar bir xil 1,920 inputda, seed/data bir xil, adaptive=False,
K=8, T=3, qiymatlar 0–63 bilan qayta o'lchandi:

| Checkpoint | Actual params | CE | Accuracy | Used circuits | Theoretical reachable | MAC/example |
|---|---:|---:|---:|---:|---:|---:|
| ne20_v12_full | 20,246,881 | 0.82627 | 73.385% | 1391/1408 | 1408 | 4,142,208 |
| ne50_v12_coverage_full | 50,327,905 | 0.84332 | 72.240% | 3536/3712 | 3712 | 4,142,208 |
| ne100_v12_coverage_full | 100,466,025 | 0.82296 | 73.750% | 7116/7552 | 7552 | 4,151,424 |

Bular **53.62/54.77/54.43% eski scaling runlarining checkpointlari emas**.
V12'da numeric encoding, slot encoder va halting supervision bor; NE20 trainingda
NE50/100 dagi coverage regularizer yo'q. Tarixiy training split `all` bo'lgan:
hash-heldout inputlarda qayta evaluation yangi train-heldout generalization
isbotlamaydi. Matched evaluation mavjud, matched training YO'Q.

100M'da ayni vaznlarda depth5 -> depth4: accuracy 73.750% -> 73.385%, ishlatilgan
circuit 7116 -> 3734. To'liq depth bankni matematik jihatdan ochgan, lekin shu
controlning sifat farqi kichik. Depthni o'zgartirish address semanticsni ham
o'zgartiradi; bu sof "coverage only" intervention emas.

Contiguous-window implementatsiyasida depth4'ning nazariy reachable circuit
soni faqat 8^4 emas: window offsetlari sabab min(E, 8^4 + M - 1). M=32 uchun
4127. Bu baribir E=7552'ni to'liq qamramaydi.

Hali ajratilmagan narsalar:

- 5000 qadam yetarliligi: teng protokolli learning curve/continuation kerak;
- active K juda kichikligi: teng training budgetli K ablation kerak;
- optimization instability: matched training seedlar kerak;
- yangi parametr hissasi: mustaqil o'qitilgan banklarda row ID'lar semantik
  jihatdan mos emas. Har scalar parametrga alohida sababiy foyda yozish bu
  checkpointlardan aniqlanmaydi. Nested/function-preserving growth va task-wise
  group ablation kerak; hozirgi per-row ablation buni almashtirmaydi.

Shu sabab katta modellarning yangi training runlari yoki 300M scaling boshlanmadi.
Bu band ochiq qolmoqda, yangi accuracy jadvali bilan "yechildi" deyilmaydi.

## P-006: nimalar aniq, nimalar estimate

- Total count requires_grad'dan mustaqil: frozen weightlar ham parametr.
- Unique active count sample bo'yicha union: takror ishlatilgan row T marta
  saqlangan parametr sifatida sanalmaydi.
- Har decision touched parametrlar shared GRU/head, router va selected rowlarni
  o'z ichiga oladi; key read multiplicity alohida.
- Router projection, key-score, pair-score, encoder/initial state, circuit body,
  GRU, memory write, output va halt head MAC komponentlari ajratilgan.
- Numeric encoder barcha T token pozitsiyalarida ishlaydi; formula T*13*d_model,
  faqat numeric tokenlar soni emas.
- Hierarchical formula configured depth emas, active_depth'ni ishlatadi.
- MAC va parameter bir narsa emas; arithmetic FLOP = 2*MAC konvensiyasi.
  Norm, nonlinear, top-k, gather, cache/DRAM va kernel xarajatlari bu MACga
  kirmaydi. Ularni exact profiler FLOP yoki physical bytes deb ko'rsatmaymiz.

Har bank-credit armda o'lchangan training forward budget:

- main: 30,000 example trajectory, 180,000 circuit-row forward,
  14,960,640,000 analytical MAC;
- probe: 7,500 trajectory, 45,000 row forward, 3,740,160,000 MAC;
- 500 main backward + 500 probe backward; training teacher forward yo'q;
- backward MAC `null`: frozen-controller input-gradient hisobini soxta 2x
  koeffitsiyent bilan aniq deb ko'rsatmadik.

Static/on-policy calibrationning har birida 4320 initial-reference va 36000
fresh-relabel example forward, jami 20,107,100,160 analytical MAC; 200 router-key
backward, body backward yo'q. Static ham fresh diagnostic xarajatini to'laydi,
lekin uni training target sifatida ishlatmaydi.

Ledger actual-update snapshoti O(bank parameters) memory/read diagnostikasidir.
U latency o'lchashdan tashqarida; bu xarajat sparse training efficiency dalili
sifatida yashirilmagan. 20M/50M/100M read-only evaluationda bunday training
snapshotlar olinmadi; tarixiy gradient/update ma'lumotlari `null`.

## Gate va yakuniy qaror

Oldindan benchmarkdagi gate: controlga nisbatan mean accuracy >=+2 pp,
hech bir seed manfiy emas, CE noninferior, har seed mean/p95 regret >=10%
yaxshilanishi, candidate recall noninferior.

Underused-margin natijasi:

- accuracy +0.104 / +0.052 pp; mean **+0.078 pp**;
- mean regret reduction 0.859% / 1.190%; p95 0.668% / 0.588%; recall o'zgarmadi.

**RAD ETILDI.** Gradient exposure oshdi, lekin foydali specialization va katta
sifat o'sishi ko'rsatilmagan. Hech qaysi default almashtirilmaydi. Diagnostic
patch foydali, experimental credit esa isbotlangan yechim emas.

## Qayta ishga tushirish

```powershell
python benchmark_sparse_credit_audit.py --steps 500 --calibration-steps 200 --eval-batches 16 --capacity --capacity-batches 16 --device cpu --output results/runs/sparse_credit_reproduction.json
python -m pytest tests/test_sparse_audit.py tests/test_instrumentation.py tests/test_forward.py tests/test_training_controls.py tests/test_router.py tests/test_optimizer.py -q
```

Raw run: `results/runs/sparse_credit_audit_20260907_s17_s18.json`.
Accounting-enriched nusxa: `results/runs/sparse_credit_audit_20260907_accounted.json`.
Enrichment faqat mavjud measured countsdan phase budget chiqaradi; quality,
regret, gate yoki training natijasini o'zgartirmaydi. Yangilangan benchmark
buni avtomatik qo'shadi; `--from-result` trainingni qayta boshlamaydi.

**41 focused test o'tdi.** Eski Transformer baseline testlaridan ikkita mavjud
PyTorch warning chiqdi; yangi modelda Transformer/attention qo'shilmagan.
