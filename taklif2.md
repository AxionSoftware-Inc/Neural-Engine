# Taklif 2 — scale-invariant routing arxitekturasi

Status: **keyingi tajriba; hali implementatsiya qilinmagan**

## Maqsad

20Mda topilgan foydali hisoblash yo‘li 100M, 300M, 700M yoki 1Bga o‘tganda
buzilmasligi kerak. Capacity oshishi global router uchun yangi raqiblar sonini
ko‘paytirib, eski foydali route’larni yo‘qotmasligi kerak.

Bu taklif `TypedRegisterNeuralEngine`ni tashlab yubormaydi. Register/dataflow
grafigi saqlanadi; faqat circuit bankka kirish usuli o‘zgartiriladi.

## Hozirgi muammo

Hozirgi sxema:

```text
query -> katta global bankdan top-k circuit
```

20Mdan 300Mga o‘tganda stored parametrlar ko‘payadi, lekin active yo‘l
taxminan 1.51M bo‘lib qoladi. Katta bankdagi circuitlar uchun gradient va route
reuse kamayadi. Bu capacity’ni capabilityga aylantirishni qiyinlashtiradi.

Shuning uchun har model hajmida routing parametrlarini qo‘lda qayta sozlash
kerak bo‘lsa, bu scalable arxitektura hisoblanmaydi.

## Taklif qilinayotgan yangi graph

```text
register state
      |
      v
stable family router(op + stage)
      |
      +--> local value router --> local top-k circuits
      |
      +--> shared fallback circuit
      |
      v
register write
```

### 1. Stable family router

Birinchi router global millionlab circuitlar orasidan bevosita tanlamaydi.
U avval operator va stage asosida logical family tanlaydi:

```text
(add, stage-1), (multiply, stage-2), (readout, stage-3), ...
```

Family soni kichik va model hajmidan mustaqil bo‘ladi. Bu routerning asosiy
qarori 20M, 100M va 300Mda bir xil semantikaga ega qoladi.

### 2. Local value router

Family tanlangandan keyin numeric value context faqat shu family ichida
ishlatiladi. Bu `typed-only` tajribasidagi haddan tashqari qattiqlikni va full
state routingidagi global fragmentationni bir vaqtda kamaytiradi.

Local router compressed value contextdan foydalanishi mumkin, lekin compression
global route identity emas, family ichidagi tanlovga xizmat qiladi.

### 3. Shared fallback

Har stage uchun kichik shared circuit yo‘li qoladi. Local router noto‘g‘ri
family/circuit tanlaganda butun computation uzilib qolmasligi kerak. Fallback
majburiy dense path emas; u active budget ichidagi himoya yo‘li bo‘ladi.

### 4. Capacity growthni route-compatible qilish

Model kattalashganda yangi circuitlar eski global bankka aralashtirilmaydi.
Ular mavjud family ichida replica sifatida qo‘shiladi:

```text
family F:  circuit-1 ... circuit-32
growth:    circuit-33 ... circuit-64  # shu family ichida
```

Oldingi foydali circuitlar parent checkpointdan ko‘chiriladi. Yangi replica
random boshlanmaydi; parent weightdan clone/split qilinadi. Shu sabab capacity
oshishi oldingi knowledge’ni buzmasligi kerak.

## Training va regularization

Arxitektura o‘zgarishi bilan quyidagi objective’lar ham kerak bo‘ladi:

- family load-balancing loss;
- local circuit usage regularization;
- route consistency/distillation loss: growthdan oldin va keyin bir xil
  calibration savolda family qarori saqlansin;
- dead circuit detection va controlled recycle;
- stage-level intermediate supervision;
- route replay: final lossga olib kelgan trajectory qayta tekshirilsin.

Bu regularizerlar circuitlarning hammasini aktivlashtirmaydi. Maqsad — faqat
kerakli active pathni ishlatish, lekin yangi capacityni foydali family ichida
joylashtirish.

## Nimalarni qilmaymiz

- attention yoki Transformer qo‘shmaymiz;
- `operator_partitioned` kabi qattiq private partitionni default qilmaymiz;
- faqat `candidate_pool` yoki `active_circuits`ni kattalashtirib muammoni
  yashirmaymiz;
- 300Mdan keyin darhol 700M/1Bga o‘tmaymiz;
- har model hajmi uchun alohida qo‘lda tuning qilib, buni scaling deb
  hisoblamaymiz.

## Amalga oshirish tartibi

### Phase A — 20M architecture control

1. Hozirgi branchdagi typed-register checkpointni baseline sifatida muzlatish.
2. Stable family + local top-k + fallback routerini implementatsiya qilish.
3. Eski 20M bilan bir xil seed, step, batch, loss va `64³` full-domain test.
4. Route reuse, family usage, dead fraction va active parameterlarni o‘lchash.

### Phase B — scale test

Bir xil router config bilan 100M va 300Mda sinash. Faqat family ichidagi
capacity/block soni o‘zgaradi; qo‘lda maxsus routing sozlamasi berilmaydi.

### Phase C — parent-based growth

100M checkpointdan 300Mga o‘tishda circuit familylar clone/split qilinadi.
Yangi circuitlar parent weightsdan boshlanadi. Keyin ayni usul 700M uchun
faqat 300M natijasi barqaror bo‘lsa sinovdan o‘tkaziladi.

## Qabul qilish mezonlari

Taklif faqat quyidagi shartlar bajarilsa ijobiy deb hisoblanadi:

1. 20M yangi router baseline’dan sezilarli yomonlashmaydi.
2. 100M yangi router 89.16% full hidden reference’ni kamida saqlaydi.
3. 300M 100M natijasidan pastga tushmaydi yoki farq statistik jihatdan
   tushuntiriladi.
4. Bir xil config bilan capacity oshganda route-family reuse saqlanadi.
5. Dead circuit fraction va route fragmentation oshib ketmaydi.
6. Active compute nazorat qilinadi; barcha bank dense aktivlashtirilmaydi.
7. Natija kamida ikki seedda takrorlanadi.

## Falsifikatsiya sharti

Agar stable family + local routing + parent growthdan keyin ham 300M 100Mdan
barqaror ravishda yomon chiqsa, muammo faqat global routerda emas. Unda
micro-circuit composition, 384-d working state yoki sparse trajectory credit
assignmentni qayta ko‘rib, typed-register core’ning o‘zini yangilash kerak
bo‘ladi.

## Bog‘liqlik va tarix

Bu taklif quyidagi kuzatuvlarga asoslangan:

- `taklif.md` 3, 4, 5, 15, 16 va 22-bo‘limlari;
- V0.28 typed-register scale audit;
- V0.29 routing specialization va seed audit;
- V0.30 active-budget audit;
- V0.31 full-domain question audit.

V0.29dagi qattiq partition va compressed routing natijalari bu taklifga
to‘g‘ridan-to‘g‘ri ko‘chirilmaydi; ular faqat nimani qilmaslik kerakligini
ko‘rsatadi. Yangi variantning farqi — stable logical family, family ichidagi
local value routing va parent-compatible capacity growth uchalasini birga
ishlatishidir.
