# Taklif 1 — bajarilgan tajribalar va rad qilingan yo‘llar

Status: **yakunlangan audit**

Bu fayl `taklif.md`dagi umumiy scale rejasidan keyin amalda bajarilgan
eksperimentlarni bir joyga yig‘adi. Maqsad — qaysi g‘oya sinalgani, natijasi
qanday bo‘lgani va nima sabab keyingi bosqichga o‘tkazilmaganini yo‘qotib
qo‘ymaslik.

## Ishlatilgan asosiy g‘oya

Transformer va attention ishlatmaydigan `TypedRegisterNeuralEngine` yaratildi.
Hisoblash grafigi:

```text
(a, b, op1) -> partial -> (partial, c, op2) -> final -> readout
```

Modelda operand, partial, final va readout registerlari bor. Circuit bank
sparse router orqali tanlanadi. Bu oddiy kattalashtirilgan recurrent model
emas, balki explicit dataflow arxitekturasi.

## Sinab ko‘rilgan tajribalar

### 1. Typed-register scale audit — qabul qilingan, lekin scaling isbotlanmagan

V0.28da 20M, 50M, 100M va 300M modellar bir xil protocol bilan sinaldi.
16³ deterministic grid natijalari:

| Model | All-pairs | Hidden-pairs |
|---|---:|---:|
| 20M | 57.67% | 65.56% |
| 50M | 56.48% | 63.34% |
| 100M | 57.93% | **89.87%** |
| 300M | 57.38% | 89.20% |

Natija: explicit register/dataflow g‘oyasi eski non-register NE controlidan
ancha yaxshi chiqdi. 100M kuchli signal berdi. Lekin 300M 100Mdan oshmadi,
shuning uchun “capacity oshsa quality majburiy oshadi” degan xulosa qilinmadi.

### 2. Direct readout — rad qilindi

100Mda routed readout o‘rniga direct readout qo‘yildi. Hidden grid accuracy
83.37% bo‘ldi; routed reference 89.87% edi.

Sabab: readout ham route qilinishi natijaga foydali bo‘ldi. Direct readout
modelning sparse execution g‘oyasini susaytirdi.

### 3. Operator-partitioned routing — default sifatida rad qilindi

Circuit bank operatorlar bo‘yicha qattiq bo‘lindi.

- 20M seed17: 87.01% hidden;
- 20M seed18: 61.46% hidden;
- 100M seed17: 68.65% hidden.

Bitta seeddagi yuqori natija takrorlanmadi. Qattiq partition routing variance’ni
oshirdi va umumiy scaling qonunini bermadi.

### 4. Typed-only router — rad qilindi

Router faqat `operator + stage`ni ko‘rdi, numeric value routingdan olib tashlandi.
20M all-pairs accuracy 53.15% bo‘ldi. Route reuse oshdi, ammo value-dependent
hisoblash uchun kerakli moslashuv yo‘qoldi. Demak routerga value context kerak,
lekin uni katta global bank bo‘ylab to‘liq yoyish ham noto‘g‘ri.

### 5. Shared/private hybrid routing — rad qilindi

Shared bank va operator-specific private banklar birga berildi.

- all-pairs: 56.49%;
- hidden-pairs: 60.82%.

Bu oddiy shared/private bo‘linishi muammoni hal qilmadi.

### 6. Compressed route context — hozircha rad qilindi

Routerga full numeric state o‘rniga kichik compressed value context berildi.

- 20M, 32-d context: 57.92% all, 71.75% hidden;
- 100M, 32-d context: 57.16% all, 61.22% hidden;
- 20M, 4-d context: 57.09% all, 60.68% hidden.

20Mda ijobiy ko‘ringan signal 100Mda saqlanmadi. Compression yo‘nalishi
keyinchalik local routing ichida ishlatilishi mumkin, ammo hozirgi global
router bilan default qilinmaydi.

### 7. Active circuits 8 -> 16 — rad qilindi

20Mda active circuitlar ikki baravar oshirildi:

- all-pairs: 57.67% -> 57.06%;
- hidden-pairs: 65.56% -> 63.35%;
- training vaqti taxminan 1.5x oshdi.

Ko‘proq active width hozirgi credit-assignment muammosini hal qilmadi.

### 8. Full-domain question audit — benchmark tuzatildi

Oldingi `16³` grid faqat `0..15` operandlarni tekshirgan. Inference-only auditda
to‘liq `64³ = 262,144` savol/operator juftlik ishlatildi.

Hidden-stage checkpointlar:

| Model | Full hidden grid |
|---|---:|
| 20M | 64.89% |
| 50M | 63.06% |
| 100M | **89.16%** |
| 300M | 85.42% |

All-pairs checkpointlar barcha 9 juftlikda 58.10%, 58.10%, 59.23% va 58.60%
berdi. Aynan bir xil ikki qiyin juftlikda all-pairs checkpointlar 60.62%,
60.67%, 62.41% va 61.40% berdi.

10 ta qo‘lda tanlangan edge/interior savolda 100M va 300M 10/10 to‘g‘ri javob
berdi. 20M va 50M bir xil bitta qiyin holatda adashdi:

```text
(63 * 63 + 63) mod 64 = 0, model prediction = 32
```

Inference deterministik chiqdi. Evaluatorga bir xil savol juftliklarini explicit
berish uchun `--pair` parametri qo‘shildi.

## Muhim metodik muammo

Oldingi “hidden” protocol haqiqiy zero-shot emas. Model avval barcha 9 operator
juftligini ko‘rgan, keyin 5,000 qadam davomida 6 ta visible juftlikda
adaptatsiya qilingan va 2 ta hidden juftlikda o‘lchangan. Shuning uchun hidden
score post-exposure adaptationni bildiradi, primitive compositionning toza
zero-shot generalizatsiyasini emas.

## Yakuniy hukm

Qabul qilingan signal:

- typed registers va explicit dataflow ishlayapti;
- attention/Transformer bu synthetic control uchun shart emas;
- 100M hozirgi eng yaxshi checkpoint;
- sparse active path taxminan 1.51M atrofida qolmoqda.

Rad qilingan yoki yetarli bo‘lmagan yo‘llar:

- faqat bankni kattalashtirish;
- qattiq operator partition;
- typed-only routing;
- shared/private hybrid;
- juda kichik compressed context;
- active circuitlarni shunchaki ko‘paytirish;
- 700M/1Bga ko‘r-ko‘rona o‘tish.

Asosiy ochiq muammo: stored capacity oshganda yangi circuitlar foydali active
hisoblashga aylanmayapti. Keyingi taklif shu routing scale muammosini
arxitekturaviy hal qilishga qaratilgan.

## Manbalar

- `taklif.md` — dastlabki umumiy scale va sparse-training taklifi;
- `results/V0_28_TYPED_REGISTER_SCALE.md`;
- `results/V0_29_ROUTING_SPECIALIZATION_AUDIT.md`;
- `results/V0_30_ACTIVE_BUDGET_AUDIT.md`;
- `results/V0_31_FULL_DOMAIN_QUESTION_AUDIT.md`.

V0.32dagi family-local router ham sinab ko‘rildi va rad qilindi. 20Mda
all-pairs 57.76%, hidden full-domain 61.37% berdi; reference mos ravishda
58.10% va 64.89% edi. Bu variant 100M/300Mga ko‘tarilmadi.

V0.33dagi role-anchored router 20M hidden full-domain accuracy’ni 71.13%ga
oshirdi, ammo 100Mda 78.32% bo‘lib, eski 100M reference 89.16%dan past qoldi.
V0.34 fixed role-cell routing 20Mda 78.28% hidden berdi, lekin all-pairs
57.93% va 100M all-pairs 57.08% bo‘ldi; ikkalasi ham scaling uchun rad qilindi.

V0.35 shared-residual circuit bank ham rad qilindi: 20M all-pairs 57.61%,
hidden 60.49%.

V0.36 multiplicative register interaction esa yangi kuchli quality reference
bo‘ldi. 20M all-pairs full 64³ grid 94.95%, hidden full grid 97.27% chiqdi.
100M ayni arxitektura 94.13% berdi, ya’ni interaction muammosi hal bo‘lgan,
ammo capacity scaling hali monoton emas. Tafsilotlar
`results/V0_36_MULTIPLICATIVE_REGISTER_WRITE.md`da; keyingi taklif
`taklif5.md`dagi factorized/reusable basis bank.

V0.37 factorized/reusable basis bank bu muammoni all-pairs nazoratida tuzatdi:
20M/100M/300M virtual scale full grid mos ravishda 96.22%/96.25%/96.40%.
Fizik parametrlar 2.60M/5.71M/12.64M, active estimate esa taxminan 1.79M
bo‘lib qoldi. Hidden-stage 20M/100M/300M 98.66%/97.15%/96.39% bo‘ldi; shu
sabab all-pairs scaling ijobiy qabul qilindi, hidden scaling esa ikkinchi seed
va curriculum bilan yana tekshirilishi kerak. Tafsilot
`results/V0_37_FACTORIZED_VIRTUAL_CAPACITY.md`da.
