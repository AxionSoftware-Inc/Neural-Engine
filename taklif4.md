# taklif4.md — Pretrained Qwen FFN circuits + learned sparse router

## Maqsad

`taklif3.md` dagi exact/near-exact FFN circuit decomposition ishlaganidan keyin, original Qwen bilimini imkon qadar saqlagan holda barcha circuitlarni ishlatishdan kichik active subsetga o‘tish.

Bu bosqich Neural Engine falsafasining real pretrained LLM ustidagi birinchi asosiy testi bo‘ladi:

```text
large stored FFN capacity
        ↓
small selected circuit subset
        ↓
Qwen qualityni maksimal saqlash
```

## Starting point

Teacher:

```text
original pretrained Qwen
```

Student:

```text
same Qwen architecture
same attention
same embeddings
FFN = taklif3 circuit decomposition
+ new sparse router
```

Dastlab studentda barcha FFN circuitlari active bo‘ladi va teacherga deyarli identik bo‘lishi kerak.

## Bosqichma-bosqich sparsification

Bir zumda 96 circuitdan 8 taga tushmang.

Tavsiya:

```text
100% active
-> 75%
-> 50%
-> 32 circuits
-> 16 circuits
-> 8 circuits
```

Har stage faqat oldingi stage quality gate’dan o‘tsa davom etadi.

## Router training

Router input sifatida quyidagilarni test qilish mumkin:

- token hidden state;
- layer ID;
- local context summary;
- optional recurrent/task state;
- later: semantic family/address signal.

Trainingda local candidate pool ichida soft/stochastic routing ishlatish mumkin, inference’da hard sparse routing.

## Distillation losses

Faqat final token CE yetarli bo‘lmasligi mumkin.

Teacher’dan bir necha darajada signal olish:

```text
1. final logits distillation
2. layer hidden-state matching
3. FFN output matching
4. optional attention-residual matching
5. original text loss
```

Suggested combined objective:

```text
L = a * LM_loss
  + b * logits_KL
  + c * hidden_match
  + d * FFN_output_match
  + e * routing_regularization
```

## Freeze strategy

Birinchi tajribada pretrained Qwen’ning katta qismini freeze qilish foydali:

```text
freeze attention
freeze embeddings
freeze most original weights
train router + limited circuit adaptation
```

Keyin kerak bo‘lsa circuit weightlarni past LR bilan unfreeze qilish.

Maqsad bilimni qayta yaratish emas, mavjud funksiyani sparse yo‘lga moslashtirish.

## Muhim metrics

Har sparsity levelda:

```text
perplexity
standard benchmark quality
logit KL vs teacher
hidden-state similarity
active FFN params/token
FLOPs/token
VRAM
latency
throughput
route entropy
circuit utilization
```

## Essential control

Random pruning va magnitude pruning control bo‘lsin.

Agar learned Neural Engine router random/magnitude baseline’dan yaxshiroq quality-efficiency curve bermasa, novelty signal zaiflashadi.

## GO / NO-GO

STRONG GO misoli:

```text
<= 1-2% relative benchmark quality loss
with >= 4x FFN active-compute reduction
```

VERY STRONG GO:

```text
near-original quality
with 8x+ FFN active-compute reduction
```

NO-GO:

```text
quality active fraction bilan keskin qulaydi
va learned routing oddiy pruningdan ustun emas
```

## Critical question

Bu tajriba quyidagi savolga bevosita javob beradi:

> Pretrained dense modeldagi stored knowledge’ning katta qismini saqlagan holda, har token uchun faqat kichik relevant FFN circuit subsetini ishlatish mumkinmi?

Agar javob ha bo‘lsa, bu scratch 7B trainingdan oldin juda katta de-risking natija bo‘ladi.

## Keyingi bosqich

Taklif4 muvaffaqiyatli bo‘lsa, `taklif5.md` dagi hybrid Qwen + Neural Engine modeliga o‘tish.
