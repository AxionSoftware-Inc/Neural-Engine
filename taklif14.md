# taklif14.md — Pretrained Qwen conversion experiments: canonical execution order and stop gates

Status: **canonical main-branch master plan**

> Branch-number collision sabab oldingi execution-plan proposalining main uchun yangi canonical raqami. Eski `taklif8.md` tarix sifatida saqlanadi.

## Maqsad

`taklif9.md` through `taklif13.md` bitta ulkan refactor sifatida implement qilinmaydi. Har bosqich mustaqil falsifikatsiya qilinadi va qimmat experiment faqat arzonroq oldingi assumption yashab qolsa boshlanadi.

## Stage 0 — current NE researchni muzlatib reference yaratish

Pretrained conversion boshlanishidan oldin current best Neural Engine checkpoint/resultlari aniq versionlanadi:

- physical trainable params;
- virtual capacity if factorized;
- active estimate;
- route reuse/utilization;
- benchmark protocol;
- seed;
- exact commit/branch.

Synthetic arithmetic natijalar real LLM capability sifatida talqin qilinmaydi; ular architecture components uchun research evidence sifatida qoladi.

## Stage 1 — `taklif9`: exact Qwen FFN decomposition

Target: smallest practical Qwen3 dense checkpoint.

```text
original Qwen
vs
all-circuits converted Qwen
```

Required:

```text
logits/perplexity ~= identical
```

STOP:

- all circuits active bo‘lsa ham material quality loss;
- algebra yoki implementation correctness noaniq.

Router trainingga exact representation tasdiqlanmaguncha o‘tilmaydi.

## Stage 2 — `taklif10`: learned FFN sparsification

Schedule:

```text
100%
-> 75%
-> 50%
-> 25%
-> 16 circuits
-> 8 circuits
```

Har pointda teacher quality, pruning baselines, latency va active compute yoziladi.

Primary gate:

```text
>=4x FFN active-compute reduction
with small quality degradation
AND learned routing > naive pruning
```

Agar foydali sparsity faqat ~1.2-1.5x bo‘lsa, deeper transplantni to‘xtatib router/circuit representation researchiga qaytiladi.

## Stage 2B — optional factorized pretrained circuit test

Faqat direct routed circuits ishlagandan keyin:

```text
Qwen circuit slices
-> shared factor basis
-> virtual circuits
```

Required controls:

- same physical parameter budget;
- same training tokens/steps;
- direct non-factorized routed student;
- quality vs active compute;
- physical params vs virtual capacity explicitly separate.

Factorization direct router failureini yashirish uchun ishlatilmaydi.

## Stage 3 — `taklif11`: hybrid Qwen + NE

First endpoint:

```text
all original attention retained
FFN = sparse routed NE circuits
```

Keyin optional persistent working memory.

So‘ng attention frequency:

```text
every layer
-> every 2nd layer
-> every 4th layer
```

Har kamaytirishda real latency/KV-cache foydasi quality loss bilan birga o‘lchanadi.

STOP when quality/runtime tradeoff worsens. Hybridning o‘zi final endpoint bo‘lishi mumkin.

## Stage 4 — `taklif12`: progressive attention transplant

```text
1 block
-> 2-4 blocks
-> 25%
-> 50%
-> majority
```

Har stage:

- local/stack matching;
- end-to-end distillation;
- language/reasoning eval;
- long-context eval;
- systems measurements.

STOP if additional replacement causes unrecoverable quality loss.

## Stage 5 — scale the method

Only after smallest checkpoint works:

```text
0.6B-class
-> 1.7B-class
-> optional 4B
-> 8B-class
```

Har size uchun:

```text
teacher quality
student quality
quality retained
physical stored params
virtual capacity if any
active params/token
FLOPs/token
transfer tokens
GPU-hours
peak VRAM
RAM
latency
throughput
```

## Equal-compute rule

Bu majburiy.

Agar grown/converted student parent checkpointdan keyin yana N training steps olsa, tegishli non-growth / teacher continued-training control ham imkon qadar ayni qo‘shimcha budgetni oladi.

Bu “extra training effect”ni “new capacity/architecture effect”dan ajratadi.

## Stage 6 — 8B proof

8B faqat oldingi size’larda favorable curve reproducible bo‘lsa qilinadi.

Asosiy savol:

> Real pretrained 8B-class model near-original useful capabilityni saqlagan holda Neural Engine materially kamroq active compute, memory traffic va runtime cost bilan ishlay oladimi?

Bu undertrained scratch 8B yaratishdan ilmiy va tijoriy jihatdan qimmatroq proof.

## Required experiment log

Har run:

```text
commit SHA
branch
teacher checkpoint
student checkpoint
seed
training tokens/steps
GPU model
GPU-hours
peak VRAM
system RAM
physical params
virtual capacity
active params
FLOPs/MAC proxy
latency
throughput
perplexity
benchmark scores
teacher-student KL
route utilization
failure notes
```

Negative results saqlanadi.

## Decision tree

```text
Taklif9 exact conversion?
  NO -> fix decomposition; STOP
  YES
   |
Taklif10 useful sparse routing?
  NO -> routing/circuit research; STOP deeper transplant
  YES
   |
Taklif11 hybrid useful?
  NO -> sparse-FFN conversion is endpoint
  YES
   |
Taklif12 attention transplant useful?
  NO -> hybrid is endpoint
  YES
   |
Scale 0.6B -> 1.7B -> 4B -> 8B
```

## Canonical numbering

Main branchda bundan keyin Qwen pretrained-conversion programi uchun canonical references:

```text
taklif9  = exact FFN circuit decomposition
taklif10 = learned sparse FFN router
taklif11 = hybrid Qwen + Neural Engine
taklif12 = progressive attention transplant
taklif13 = full knowledge-transfer roadmap
taklif14 = execution order / stop gates
```

Oldingi `taklif3`-`taklif8` nusxalari tarix sifatida qoladi; yangi ish va agent instructions `taklif9`-`taklif14`ga reference qilishi kerak.