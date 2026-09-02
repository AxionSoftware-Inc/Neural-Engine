# taklif17.md — Capability frontier, OOD and anti-saturation scaling protocol

Status: **evaluation proposal; required before larger quality-scaling claims**

## Muammo

Current arithmetic benchmark 500M/700M virtual capacityda ~99.66% atrofida saturationga keldi. Bunday ceiling ostida 700M -> 1B qilish modelning capabilitysi oshganini ko‘rsatmaydi.

Shuningdek range-shift OOD (`train 0–31`, `test 32–63`) juda past bo‘ldi. Demak in-distribution composition va haqiqiy representation generalization alohida o‘lchanishi kerak.

## Asosiy savol

> Model hajmi/capacity oshgani sari kichik model qila olmaydigan yangi computation classlarini kattaroq model bajara boshlayaptimi?

Bu accuracy-at-ceiling emas, **capability frontier** o‘lchovi.

## Difficulty ladder

Bir xil primitive vocabulary bilan bosqichma-bosqich:

```text
2-step fixed graph
-> 4-step
-> 6-step
-> 8-step
-> variable-length program
-> dynamic read/write registers
-> unseen program templates
-> unseen operator combinations
```

Har difficulty uchun 20M/100M/300M/500M/700M (yoki mavjud mos scale) bir xil protocolda o‘lchanadi.

## OOD axes

Kamida alohida:

```text
A. held-out operand combinations, all values seen
B. unseen value ranges
C. held-out operation order/program templates
D. longer-than-training program length
E. noisy/distractor inputs
F. optional new primitive/operator transfer
```

Bir OOD natijani boshqasiga umumlashtirmang.

## Primary metric

Faqat accuracy emas:

```text
maximum task difficulty at >= target accuracy
```

Masalan:

```text
20M  : 4-step @ >=90%
100M : 6-step @ >=90%
300M : 8-step @ >=90%
```

shaklidagi curve haqiqiy capacity->capability signal bo‘ladi.

## Scaling controls

Har size uchun:

- same data distribution;
- same training-token/step budget yoki compute-normalized parallel report;
- multiple seeds;
- physical params, virtual capacity va active params alohida;
- scratch vs parent-growth alohida;
- matched dense Transformer baseline kamida muhim checkpointsda;
- simple recurrent/MLP baseline where appropriate.

## Anti-cheating rules

- task graphni architecturega haddan tashqari qo‘lda encode qilib bermaslik;
- hidden/OOD protocol training tarixida oldindan exposure olmaganini tekshirish;
- deterministic exhaustive grid mumkin bo‘lsa ishlatish;
- benchmark ceilingga yetsa keyingi difficultyga o‘tish;
- 99.x%ni intelligence-equivalent metric sifatida talqin qilmaslik.

## Router/circuit diagnostics

Har difficultyda:

```text
route reuse
circuit/factor utilization
dead capacity
active path length
route entropy
counterfactual route sensitivity
register causal probes
```

Quality growth yangi useful routes/cells bilan bog‘liqmi yoki faqat memorizationmi ajratiladi.

## GO criterion

Strong scaling evidence:

```text
larger stored/virtual capacity
+ approximately bounded active compute
+ reproducible increase in maximum solvable difficulty
+ gains survive OOD/composition controls
```

## NO-GO

Agar capacity 20M -> 700M oshsa-yu:

- maximum solvable difficulty siljimasa;
- faqat benchmark memorization yaxshilansa;
- OOD o‘zgarmasa;
- active compute keskin oshmasdan gain chiqmasa,

unda current architecture uchun quality scaling claim to‘xtatiladi.

## Maqsad

Bu protocol 500M/700M `99.66%` natijasini rad qilmaydi. U natija shu fixed task solved ekanini bildiradi. Ushbu taklif esa keyingi savolni o‘lchaydi: **kattaroq model haqiqatan kuchliroqmi?**
