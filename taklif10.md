# taklif10.md — Pretrained Qwen FFN circuits + learned sparse router

Status: **canonical main-branch proposal; first real pretrained sparsity test**

> Branch-number collision sabab oldingi Qwen sparse-router proposalining main uchun yangi canonical raqami. Eski `taklif4.md` tarix sifatida saqlanadi.

## Maqsad

`taklif9.md` exact/near-exact FFN circuit decomposition ishlaganidan keyin, original Qwen bilimini imkon qadar saqlagan holda har token uchun faqat kichik relevant circuit subsetini ishlatish.

```text
large pretrained FFN capacity
        ↓
learned sparse router
        ↓
small active circuit subset
        ↓
near-original Qwen quality
```

Bu Neural Engine falsafasining real pretrained LLM ustidagi birinchi asosiy testi.

## Starting point

Teacher:

```text
original pretrained Qwen
```

Student:

```text
same embeddings
same attention
same layer norms
FFN = taklif9 exact circuit decomposition
+ learned sparse router
```

Dastlab studentda barcha FFN circuitlari active va teacherga deyarli identik bo‘lishi kerak.

## Bosqichma-bosqich sparsification

Bir zumda 100% dan 8 circuitga tushmang.

```text
100% active
-> 75%
-> 50%
-> 25%
-> 16 circuits
-> 8 circuits
```

Har stage oldingi checkpointdan davom etadi va quality gate’dan o‘tmasa keyingi sparsityga tushilmaydi.

## Router input variantlari

Minimaldan boshlash:

- current token hidden state;
- layer ID / layer embedding;
- optional local context summary;
- optional recurrent/task state.

Keyin kerak bo‘lsa:

- semantic family signal;
- factorized address signal;
- short history of previous routes.

Inference hard sparse bo‘lishi kerak. Trainingda differentiable/stochastic local selection ishlatilishi mumkin.

## Distillation objective

Faqat final CE yetarli bo‘lmasligi mumkin.

```text
L = a * LM_loss
  + b * logits_KL
  + c * hidden_state_match
  + d * FFN_output_match
  + e * routing_regularization
```

Teacher signalini bir necha qatlamda ishlatish kerak:

1. final logits;
2. selected hidden states;
3. individual FFN output;
4. original text loss;
5. route utilization / entropy regularization.

## Freeze strategy

Birinchi experimentda:

```text
freeze embeddings
freeze attention
freeze most original Qwen weights
train router + very limited circuit adaptation
```

Agar kerak bo‘lsa circuit weightsni keyin juda past LR bilan unfreeze qilish.

Maqsad knowledge’ni qayta yaratish emas, mavjud functionni sparse yo‘lga moslashtirish.

## Essential baselines

Learned routing albatta quyidagilar bilan bir xil active budgetda solishtiriladi:

- random circuit pruning;
- magnitude-based pruning;
- fixed top-activation neuron groups;
- simple static per-layer subset;
- conventional structured pruning baseline if practical.

Agar learned router oddiy pruningdan yaxshi quality/compute curve bermasa, NE novelty va utility signali zaiflashadi.

## Factorized-bank extension

Hozirgi NE factorized virtual capacity natijasidan kelib chiqib, exact Qwen circuitlarini keyinchalik shared factorsga distill qilish alohida ablation sifatida sinanishi mumkin:

```text
pretrained Qwen circuit slices
-> factor basis
-> virtual circuits
-> sparse route
```

Bu birinchi router testiga aralashtirilmaydi. Avval direct circuit routing ishlashi kerak.

## Metrics

Har sparsity levelda:

```text
perplexity
standard language benchmarks
reasoning/math/code benchmarks
logit KL vs teacher
hidden-state similarity
active FFN params/token
FLOPs/MAC proxy
VRAM
RAM traffic
latency
throughput
route entropy
circuit utilization
dead-circuit ratio
```

## GO / NO-GO

STRONG GO misoli:

```text
<= 1-2% relative quality loss
AND
>= 4x FFN active-compute reduction
AND
learned routing clearly beats naive pruning controls
```

VERY STRONG GO:

```text
near-original quality
with 8x+ FFN active-compute reduction
```

NO-GO:

```text
quality active fraction bilan keskin qulaydi
OR
learned routing simple pruningdan ustun emas
```

## Critical question

> Pretrained dense modeldagi stored knowledge’ning katta qismini saqlagan holda, har token uchun faqat kichik relevant FFN circuit subsetini ishlatish mumkinmi?

Agar javob ha bo‘lsa, scratch 7B/8B pretrainingdan oldin juda katta de-risking natija olinadi.

## Keyingi bosqich

GO bo‘lsa `taklif11.md` — Hybrid Qwen + Neural Engine.