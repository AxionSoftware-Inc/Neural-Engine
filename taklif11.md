# taklif11.md — Hybrid Qwen + Neural Engine architecture

Status: **canonical main-branch proposal; safe commercial midpoint**

> Branch-number collision sabab oldingi hybrid proposalning main uchun yangi canonical raqami. Eski `taklif5.md` tarix sifatida saqlanadi.

## Maqsad

Dense Transformer’ni birdan to‘liq almashtirish o‘rniga, pretrained Qwen knowledge va sequence-processing qobiliyatini saqlagan holda Neural Engine sparse circuits va working-memory mexanizmlarini hybrid modelga kiritish.

## Starting point

Ideal boshlanish nuqtasi `taklif10.md`dan chiqqan sparse-FFN Qwen checkpoint.

Minimal hybrid:

```text
Embedding
-> original attention
-> sparse NE FFN circuits
-> original attention
-> sparse NE FFN circuits
-> ...
-> output
```

Bu bosqichda attention barcha layerlarda qolishi mumkin. Faqat dense FFN compute sparse circuit computationga o‘tadi.

## Keyingi variantlar

Variant A — minimal hybrid:

```text
all attention retained
all FFN -> sparse routed circuits
```

Variant B — attention frequency 2x kamaytirish:

```text
attention
NE-only block
attention
NE-only block
...
```

Variant C — har 4 blockdan bittasida attention.

Attentionni birdan deyarli yo‘qotmang. Har kamaytirish alohida benchmark va systems gate bilan amalga oshiriladi.

## Working memory qo‘shish

Hybridning keyingi bosqichida persistent memory/register state parallel olib boriladi:

```text
token hidden states
        |
        +-> sparse NE memory read/write
        |
        +-> occasional attention
        |
        +-> sparse routed circuits
```

Initial simple form:

```text
4-8 memory slots
slot dimension 256-1024
sparse read/write
fixed small active slot budget
```

Memoryning foydasi causal ablation bilan tekshiriladi: memoryni shuffle/zero/freeze qilganda qaysi capability tushishini o‘lchash kerak.

## Nega hybrid muhim

Attention global/context communication uchun kuchli. NE selective computation, reusable circuits va persistent intermediate state uchun foydali bo‘lishi mumkin.

```text
attention = global communication
NE circuits = sparse transformation/reasoning
working memory = persistent intermediate state
```

To‘liq Transformer-free bo‘lish shart emas. Agar hybrid quality/latency/memory bo‘yicha eng yaxshi nuqta bo‘lsa, u final product bo‘lishi mumkin.

## Distillation

Teacher original Qwen.

Har attention-removal stage’da:

- logits KL;
- selected hidden-layer matching;
- perplexity;
- math/code/reasoning;
- long-context retrieval;
- instruction following;
- latency/KV-cache/VRAM.

## Systems metrics

Faqat total params emas:

```text
active params/token
FLOPs/token
memory bandwidth
KV-cache size
persistent NE-state size
latency
throughput
VRAM
RAM
energy if measurable
```

Agar attention chastotasi kamaytirilsa KV cache ham kamayishi mumkin; local inference uchun bu katta foyda bo‘lishi mumkin.

## GO / NO-GO

Minimal hybrid GO:

```text
sparse FFN + original attention
near-original Qwen quality
material FFN compute reduction
```

Attention-frequency STRONG GO:

```text
>= 2x fewer attention blocks
with small quality loss
AND real latency/KV benefit
```

NO-GO:

```text
attention kamayishi bilan language/long-context quality tez qulaydi
AND NE memory bu funksiyani qoplay olmaydi
```

NO-GO full transplantni majburlash degani emas. Minimal sparse-FFN hybridning o‘zi mustaqil endpoint bo‘lishi mumkin.

## Keyingi bosqich

Hybrid barqaror bo‘lsa `taklif12.md` — progressive attention -> NE transplant.