# taklif13.md — Full pretrained Qwen -> Neural Engine knowledge-transfer roadmap

Status: **canonical main-branch roadmap**

> Branch-number collision sabab oldingi full-transfer roadmapning main uchun yangi canonical raqami. Eski `taklif7.md` tarix sifatida saqlanadi.

## Maqsad

Qwen kabi kuchli pretrained modelni scratchdan qayta pretrain qilmasdan, uning knowledge, language ability va reasoning behaviorini Neural Engine architecture’ga imkon qadar arzon ko‘chirish.

Bu roadmap `taklif9`-`taklif12` bosqichlarini bitta scaling programga birlashtiradi.

## Fundamental prinsip

Pretrained checkpointni shunchaki weight fayli emas, juda qimmat teacher/function deb qaraymiz.

```text
Qwen pretrained function
        ↓
exact FFN circuit decomposition
        ↓
learned sparse FFN routing
        ↓
hybrid architecture
        ↓
progressive attention transplant
        ↓
Neural Engine student
```

Maqsad original ulkan pretrainingni takrorlash emas. Maqsad teacher’da allaqachon yig‘ilgan bilimni ancha kam transfer-training bilan yangi architecture’ga ko‘chirish.

## Model scaling ladder

Birdan 8B bilan boshlamang.

```text
0.6B-class
-> 1.7B-class
-> optional 4B
-> 8B-class
```

Har size keyingi size uchun engineering va science riskini kamaytiradi.

## Phase A — exact structural reuse

`taklif9.md`

```text
Qwen FFN weights
-> exact circuit slices
-> all circuits active
```

Goal: zero/near-zero quality change.

## Phase B — sparse capacity conversion

`taklif10.md`

```text
all circuits active
-> learned router
-> progressively fewer active circuits
```

Goal: pretrained qualityni saqlab FFN active compute’ni material kamaytirish; >=4x kuchli gate, 8x+ very strong.

## Phase C — hybrid model

`taklif11.md`

```text
Qwen attention retained
+ sparse NE circuits
+ optional persistent memory
```

Bu final product bo‘lishi ham mumkin. Full Transformer removal research maqsadi, product talabi emas.

## Phase D — progressive attention transplant

`taklif12.md`

```text
original attention blocks
-> one/few at a time NE memory/mixing blocks
```

Goal: sequence-processing va reasoning functionlarini boshqa state/mixing architecture’ga ko‘chirish.

## Phase E — current Neural Engine discoveries bilan integratsiya

Current NE researchdan foydali komponentlar faqat alohida ablation bilan kiritiladi:

- typed/persistent registers;
- multiplicative/bilinear interactions where justified;
- factorized reusable circuit basis;
- parent-growth/warm-start capacity expansion;
- sparse hard inference;
- route reuse and dead-circuit audits.

Muhim: synthetic arithmetic benchmarkda ishlagan inductive biasni real language modelga avtomatik ko‘chirmang. Har komponent Qwen teacher against ablation bilan isbotlanadi.

## Factorized pretrained conversion hypothesis

Direct Qwen circuit routing ishlagandan keyin kuchli optional experiment:

```text
Qwen exact circuits
        ↓ distillation
shared factor basis
        ↓
large virtual circuit address space
        ↓
sparse active composed circuits
```

Bu physical parameter count, virtual capacity va active compute’ni alohida o‘lchashni talab qiladi. “Virtual 500M” kabi nomlar physical trainable params bilan aralashtirilmaydi.

## Distillation data

Mix:

```text
real text corpus
code
math
reasoning
instruction data
long-context samples
teacher logits/hidden states
teacher-generated hard examples
```

Teacher faqat label manbai emas, hard-example/curriculum generator ham bo‘lishi mumkin.

## Knowledge preservation metrics

Language:
- perplexity;
- factual/commonsense QA.

Reasoning:
- math;
- code;
- multi-step symbolic/compositional tests.

Instruction:
- instruction following;
- tool-use behavior where applicable.

Long context:
- retrieval;
- needle/document QA.

Systems:
- physical stored params;
- virtual capacity if factorized;
- active params/token;
- FLOPs/token;
- memory bandwidth;
- KV/state memory;
- latency;
- throughput;
- RAM/VRAM.

## Transfer-efficiency metrics

Alohida saqlash:

```text
quality retained / transfer token
quality retained / GPU-hour
quality retained / joule if measurable
```

Bu scratch pretraining bilan iqtisodiy farqni ko‘rsatadi.

## Critical baselines

Har size uchun kamida:

1. original Qwen teacher;
2. same-size continued-training control;
3. random/magnitude/structured pruning;
4. conventional sparse/MoE upcycling baseline if feasible;
5. NE converted student.

Equal-compute control majburiy: studentga qo‘shimcha training berilsa teacher/base control ham shu qo‘shimcha training budgetini olishi kerak.

## 8B gate

8B faqat oldingi size’larda quyidagilar reproducible bo‘lsa:

```text
- exact FFN conversion works
- learned routing materially reduces active compute
- useful sparsityda quality survives
- hybrid/transplant tiny modeldan kattaroqqa scale qiladi
- transfer cost scratch-equivalent pretrainingdan ancha arzon
```

## Strong 8B success definition

Illustrative, sacred threshold emas:

```text
>=95% useful teacher benchmark quality retained
AND >=3-5x active-compute reduction
AND real hardware memory/runtime gain
AND transfer cost dramatically below scratch pretraining
```

Near-original quality bilan >=8x active reduction juda kuchli result bo‘ladi.

## Failure interpretation

Full transplant ishlamasa ham partial success mustaqil qiymatga ega:

- `taklif9`: exact pretrained circuit representation;
- `taklif10`: sparse pretrained FFN conversion;
- `taklif11`: commercial hybrid architecture;
- `taklif12`: deeper architecture transplant.

## Keyingi fayl

Execution order va stop gates: `taklif14.md`.