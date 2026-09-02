# taklif12.md — Progressive Transformer attention -> Neural Engine transplant

Status: **canonical main-branch proposal; high-risk deeper transplant**

> Branch-number collision sabab oldingi attention-transplant proposalining main uchun yangi canonical raqami. Eski `taklif6.md` tarix sifatida saqlanadi.

## Maqsad

Pretrained Qwen modelidagi attention/mixing funksiyasini birdan tashlab yubormasdan, layer-by-layer Neural Engine working-memory/mixing bloklariga ko‘chirish.

Asosiy savol:

> Transformer sequence-mixing funksiyasini teacher yordamida boshqa architecture’ga bosqichma-bosqich ko‘chirib, pretrained knowledge va behaviorni katta darajada saqlash mumkinmi?

## Starting point

Ideal boshlanish nuqtasi:

- `taklif10.md` orqali FFN sparse circuitsga o‘tgan model; yoki
- `taklif11.md` hybrid model.

Original Qwen teacher doim freeze reference sifatida saqlanadi.

## Progressive replacement

Masalan N-layer modelda:

```text
Stage 0: barcha original attention
Stage 1: 1 NE memory/mixing block
Stage 2: 2-4 NE blocks
Stage 3: ~25% replaced
Stage 4: ~50% replaced
Stage 5: majority replaced
```

Har stage alohida checkpoint, eval va systems report bilan freeze qilinadi. Stage N quality gate’dan o‘tmasa N+1 qilinmaydi.

## NE replacement block

Replacement block attentionni bit-for-bit nusxalashga majbur emas. U teacher stackning funksional natijasini yaqinlashtirishi kerak.

```text
input hidden state
+ persistent memory/registers
+ sparse routed circuits
-> token residual/output
+ updated memory
```

Possible minimal structure:

```text
token/local state
   ↓
cheap address/summary
   ↓
read selected slots
   ↓
selected circuit micro-program
   ↓
write slots + token residual
```

## Distillation ladder

### Level 1 — local block matching

Bitta original attention blockni teacher qilib input-output mappingni match qilish.

```text
hidden MSE/cosine
residual-output match
optional projected feature match
```

### Level 2 — short-stack matching

2-4 consecutive teacher blocks outputini student stack bilan match qilish. Studentga exact internal imitationdan ko‘ra boshqa representation tanlash erkinligi beriladi.

### Level 3 — end-to-end distillation

```text
LM loss
+ teacher logits KL
+ selected hidden-state matches
```

### Level 4 — limited capability recovery

Math, code, reasoning, instruction va long-context data bilan controlled continued training.

## Replacement order ablation

Kamida quyidagilarni solishtirish:

1. middle layers first;
2. every Nth layer;
3. late layers first;
4. sensitivity analysis bo‘yicha least-critical blocks first.

## Critical controls

- same continued-training budget bilan original Qwen control;
- random attention removal control;
- `taklif11` hybrid attention-frequency baseline;
- same active-compute budgetdagi smaller Transformer baseline if practical.

## Metrics

Har replacement fractionda:

```text
perplexity
logit KL
hidden similarity
language benchmarks
reasoning/math/code
instruction following
long-context retrieval
active params/token
FLOPs/token
KV cache
persistent state memory
VRAM/RAM
latency
throughput
```

## GO / NO-GO

STRONG GO:

```text
25-50% attention blocks replaced
small quality loss
material latency/KV/compute gain
```

VERY STRONG GO:

```text
majority attention replaced
teacher quality largely retained
NE state takes over meaningful sequence/reasoning work
```

NO-GO:

```text
first few replacements already cause large unrecoverable quality loss
```

NO-GO bo‘lsa full Transformer-free target majburlanmaydi; `taklif11` hybrid endpoint sifatida saqlanadi.

## Keyingi bosqich

Muvaffaqiyatli bo‘lsa `taklif13.md` — full pretrained Qwen -> Neural Engine transfer/scaling roadmap.