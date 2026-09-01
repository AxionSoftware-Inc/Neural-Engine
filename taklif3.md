# Taklif 3 — role-anchor global bank routing

Status: **keyingi tajriba; hali implementatsiya qilinmagan**

## Nega taklif 2 emas

V0.32dagi stable family/local router operator-stage bo‘yicha qattiq bank
bo‘linishi sabab 20M baseline’dan past chiqdi. Har operatorga alohida tor bank
berish value composition uchun yetarli moslashuv qoldirmadi.

Endi physical bankni familylarga qattiq bo‘lmaymiz. Barcha circuitlar bitta
global bankda qoladi, lekin routingning coarse tree qismi value query bilan
fragmentatsiyalanmaydi.

## Yangi graph

```text
register query
     |
     +--> role anchor tree(op + stage) --> stable coarse cell
     |
     +--> value query(full register state)
                         |
                         v
              candidate scoring inside coarse cell
                         |
                         v
                       top-k circuit
                         |
                         v
                    register write
```

## Asosiy mexanizm

1. `role anchor tree` faqat `operator + stage` signalidan foydalanadi. Shu
   sabab bir xil primitive role turli operandlarda bir xil coarse route
   family/cellni qayta ishlatadi.
2. `value query` to‘liq numeric register state bo‘lib qoladi. U candidate
   window ichida circuitlarni score qiladi. Shu bilan V0.29dagi `typed-only`
   xatosi takrorlanmaydi.
3. Physical bank qattiq operator partitionga bo‘linmaydi. Bir coarse cellga
   turli operatorlar tushishi va shared computationdan foydalanishi mumkin.
4. Capacity oshganda yangi circuitlar coarse cell’lar bo‘yicha replica sifatida
   taqsimlanadi. Eski cell ma’nosi saqlanadi; global route tree qaytadan
   ma’nosiz raqobatga tushmaydi.

## Nima o‘zgaradi

`HierarchicalRouter` yoki uning yangi varianti ikkita query qabul qiladi:

```text
tree_state = operation_embedding + stage_embedding
value_state = full routed register query
```

Tree logits `tree_state` bilan hisoblanadi. Candidate key logits esa
`value_state` bilan hisoblanadi. Hozirgi routerda bu ikkalasi bir xil query edi;
asosiy arxitektura o‘zgarishi shu separation.

Stable tree depth va coarse-cell layout model hajmiga qarab o‘zgarmaydi. Katta
bank cell ichiga qo‘shimcha replica qo‘shadi, lekin 20M route ma’nosini 100Mda
almashtirmaydi.

## Training regularization

- coarse route consistency: bir xil operator-stage uchun tree route barqaror
  bo‘lsin;
- cell load-balancing: ba’zi cell’lar o‘lib qolmasin;
- candidate value score uchun full final-loss gradient;
- parent growthda eski tree va cell assignmentni distillation bilan saqlash;
- faqat ishlatilgan candidate circuit rows uchun lazy optimizer update.

## Sinov tartibi

### Phase A — 20M

- baseline: `exp/typed-register-composition`dagi 20M reference;
- yangi variant: ayni seed, step, batch, optimizer va loss;
- all-pairs va hidden-stage train;
- full `64³` deterministic grid;
- route-cell reuse, used/dead rows, active params va training time.

### Phase B — scale

Faqat 20M baseline’dan kamida teng natija chiqsa, shu bir xil config bilan
100M va 300Mga o‘tiladi. Model hajmiga qarab qo‘lda route tuning qilinmaydi.

### Phase C — growth

100Mdan 300Mga o‘tishda coarse tree va cell identity saqlanadi. Yangi rows
parent circuitlardan clone/split qilinadi.

## Qabul qilish mezonlari

1. 20M all-pairs `58.10%` reference’dan sezilarli pastlamasligi.
2. 20M hidden full-domain `64.89%` reference’dan sezilarli pastlamasligi.
3. 100M va 300Mda route-cell reuse capacity bilan qulamasligi.
4. 300M 100Mdan yomonlashsa, sabab route fragmentation emasligi ko‘rsatilishi.
5. Active path sparse qolishi va barcha bank dense ishlatilmasligi.

## Falsifikatsiya

Agar role/value query separation ham 20Mda baseline’ni tiklamasa, routing
muammosi emas, micro-circuit yoki fixed 384-d working state fundamental
bottleneck bo‘lishi mumkin. Unda keyingi yo‘l route tuning emas, structured
multi-slot memory yoki shared-basis circuit arxitekturasi bo‘ladi.
