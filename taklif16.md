# taklif16.md — Transformer -> Neural Engine cross-architecture parent transplant

Status: **new transfer hypothesis; not ordinary parent growth**

## Maqsad

Trained Transformer/Qwen modelini faqat teacher sifatida emas, Neural Engine student uchun function-level parent sifatida ishlatish.

Oddiy parent-growthda parent va child bir xil arxitektura oilasida bo‘ladi. Bu taklifda esa architecture o‘zgaradi:

```text
Transformer parent
      ↓
hybrid transition model
      ↓
Neural Engine child
```

Shuning uchun weightlarni to‘g‘ridan-to‘g‘ri birma-bir ko‘chirish talab qilinmaydi. Maqsad parentning funksiyasi va capabilitysini bosqichma-bosqich yangi computation graphga ko‘chirish.

## Asosiy prinsip

Birdan:

```text
Transformer OFF
NE ON
```

qilinmaydi.

Progressive morphing:

```text
Stage 0: 100% Transformer, 0% NE contribution
Stage 1: Transformer + zero/small NE residual path
Stage 2: Transformer frozen, NE local outputsni imitate qiladi
Stage 3: mixed contribution / gated handoff
Stage 4: selected Transformer blocks bypass qilinadi
Stage 5: NE selected functionsni mustaqil bajaradi
```

## Minimal experiment

Qwen3-0.6B’dan boshlash.

1. Early sensitive layersni tegmasdan qoldirish.
2. Late yoki middle bitta FFN/mixing block tanlash.
3. Parallel NE adapter/register/circuit path qo‘shish.
4. Parent outputni local distillation bilan match qilish.
5. Parent branch contributionini asta pasaytirish.
6. Parent branch 0 ga yaqinlashganda full-model fidelityni tekshirish.
7. Keyin 2–4 blockga kengaytirish.

## Handoff gate

Conceptual:

```text
y = alpha * ParentBlock(x)
  + (1-alpha) * NEBlock(x, memory)
```

Training davomida `alpha` 1.0 dan 0.0 tomon tushiriladi. Schedule fixed yoki performance-gated bo‘lishi mumkin.

Bu formula final architecture shart emas; u transition scaffold.

## Knowledge preservation losses

```text
local block output match
short-stack hidden match
teacher logits KL
original LM loss
optional register-state consistency
```

## Equal-compute control

Majburiy:

Agar cross-architecture student N additional training token/step olsa, original Qwen continued-training yoki matched adapter control ham o‘sha budgetga yaqin training olishi kerak.

Shunda gain extra trainingdanmi yoki architecture transferdanmi ajratiladi.

## Success ladder

```text
1 block transferable
-> 2–4 blocks
-> 25% stack
-> 50% stack
-> majority
```

Har stage alohida GO/NO-GO.

## STRONG GO

Misol:

```text
25–50% target compute blocks
NE pathga to‘liq handoff
small language/reasoning quality loss
material active-compute/KV/memory advantage
```

## NO-GO

- parent branchni kamaytirganda quality recover qilinmasa;
- student faqat parent parallel turganda ishlasa;
- transfer training scratch-equivalent darajada qimmatlashsa;
- real runtime advantage bo‘lmasa.

## Muhim farq

Bu `taklif12`dagi oddiy progressive attention replacementdan kengroq. `taklif12` ma’lum blocklarni distill qilib almashtirishga qaratilgan; bu taklif esa **cross-architecture parent-growth/morphing protocolini** umumiy mechanism sifatida tekshiradi va FFN, mixing, registers hamda working memoryga qo‘llanishi mumkin.

## Macro-cell bilan aloqa

Macro-cell keyin tanlansa, u NE child blockining bir varianti bo‘lishi mumkin. Ammo bu taklif macro-cellga bog‘liq emas va undan oldin kichik existing NE blocks bilan falsifikatsiya qilinishi mumkin.
