# taklif9.md — Qwen FFN -> Neural Engine exact circuit decomposition

Status: **canonical main-branch proposal; cheap first gate**

> Branch-number collision sabab oldingi Qwen proposalining main uchun yangi canonical raqami. Eski `taklif3.md` tarix sifatida saqlanadi.

## Maqsad

Pretrained Qwen modelidagi bilimni noldan qayta o‘qitmasdan Neural Engine circuit representationiga o‘tkazish mumkinligini eng arzon va eng toza usulda tekshirish.

Bu bosqichda attention, embeddings, layer norm va qolgan Transformer qismlari o‘zgarmaydi. Faqat FFN/SwiGLU intermediate dimension matematik jihatdan ekvivalent circuit guruhlariga ajratiladi.

## Asosiy g‘oya

Soddalashtirilgan FFN:

```text
x -> gate/up projections -> nonlinear hidden units -> down projection -> y
```

Intermediate hidden dimensionni bloklarga bo‘lamiz:

```text
C1 = units 0..127
C2 = units 128..255
...
Cn = remaining units

FFN(x) = C1(x) + C2(x) + ... + Cn(x)
```

Har circuit yangi random parametr emas. U original Qwen weightlarining aynan slice’laridan tuziladi. Barcha circuitlar active bo‘lganda yig‘indi original FFN outputini floating-point tolerance ichida qayta tiklashi kerak.

## Nega bu muhim

Agar exact/near-exact conversion ishlasa:

- pretrained knowledge tashlab yuborilmaydi;
- Qwen FFN capacitysi circuit-bank ko‘rinishiga o‘tadi;
- keyingi bosqichda learned sparse router qo‘shish mumkin;
- scratch pretrainingdan oldin real pretrained LLM ustida NE falsafasi sinovga tayyor bo‘ladi.

## Birinchi target

Birdan 8B bilan boshlamang.

```text
Qwen3 0.6B-class dense checkpoint
-> 1 layer correctness
-> all FFN layers correctness
-> full-model logits/perplexity check
```

Birinchi tajriba capability emas, algebra va implementation correctness haqida.

## Implementatsiya

1. Original Qwen checkpointni freeze teacher sifatida yuklash.
2. Har FFN layerning gate/up/down projectionlarini aniqlash.
3. Intermediate dimensionni 32/64/128/256 neuronlik fixed chunklarga bo‘lish.
4. Har chunk original weight slice’ini aynan meros oladi.
5. Barcha circuitlar active bo‘lgan holda layer outputni original FFN bilan solishtirish.
6. So‘ng barcha FFN layerlarni converted representationga almashtirish.
7. Full-model logits, perplexity va deterministic generationni solishtirish.

## Required correctness tests

Har layer:

```text
max_abs_error
mean_abs_error
relative_error
cosine_similarity
```

Full model:

```text
same prompt -> teacher logits vs converted logits
teacher/student KL
perplexity delta
deterministic next-token agreement
```

FP16/BF16 summation orderi sabab bit-for-bit tenglik shart emas; numerical tolerance va functional equivalence asosiy mezon.

## Optional follow-up: neuron clustering

Exact fixed chunks ishlagandan keyingina neuronlarni behavior/activation/weight similarity bo‘yicha cluster qilish mumkin.

```text
Qwen neurons
-> activation/weight statistics
-> learned clusters
-> circuit families
```

Bu exact decompositiondan alohida experiment bo‘lishi shart, chunki clustering quality loss keltirishi mumkin.

## Factorized Neural Engine bilan bog‘lanish

Hozirgi NE factorized virtual circuit banki Qwen conversion uchun qiziq keyingi yo‘l beradi:

```text
exact Qwen circuit slices
-> distill/approximate into reusable factor basis
-> many virtual circuits from shared factors
```

Lekin bu `taklif9`ning GO mezoni emas. Avval exact representation isbotlanadi; factorization keyin ablation sifatida qo‘shiladi.

## GO / NO-GO

STRONG GO:

```text
full-model logits difference ~= numerical noise
perplexity delta negligible
no meaningful benchmark degradation
```

WEAK GO:

```text
small stable numerical difference
quality materially unchanged
```

NO-GO:

```text
all circuits active bo‘lsa ham material quality loss
```

NO-GO bo‘lsa router yoki sparsificationga o‘tilmaydi; decomposition algebra/implementation tuzatiladi.

## Nima claim qilinmaydi

Bu bosqich sparse inference emas. Barcha circuitlar active.

Maqsad faqat:

> pretrained Qwen FFN funksiyasini Neural Engine circuit representationiga lossless yoki near-lossless ko‘chirish.

## Keyingi bosqich

GO bo‘lsa `taklif10.md` — pretrained FFN circuits + learned sparse router.