# Taklif 5 — factorized capacity va scale-invariant routing

Status: **ijobiy 20M–300M; 500Mda yangi route kerak**

## Kuzatuv

V0.36da multiplicative register interaction 20M full-domain accuracy’ni
58.10%dan 94.95%ga olib chiqdi. Hidden grid 97.27% bo‘ldi. Demak asosiy
muammo operandlar orasidagi interactionni ifodalashda bo‘lgan.

Ammo ayni arxitekturani 100Mga ko‘tarish 94.13% berdi. Active estimate deyarli
o‘sha 1.69M bo‘lib qoldi, active fraction esa 8.44%dan 1.63%ga tushdi. Katta
bankdagi qo‘shimcha rows ko‘proq saqlangan capacity bo‘lib qoldi, real
hisoblashga muntazam qo‘shilmadi.

## Taklif

Circuit bankni N ta mustaqil row sifatida emas, bir nechta qayta ishlatiladigan
basis komponentlari va kichik route-code kombinatsiyasi sifatida ifodalash:

```text
route query
   -> scale-invariant address / factor IDs
   -> selected basis components
   -> multiplicative pair query + composed register write
```

Har bir sample faqat tanlangan basislar va ularning kichik adapterlarini
aktivlashtiradi. Basislar ko‘p virtual circuitlar orasida qayta ishlatilgani
uchun yangi capacity qo‘shilganda har bir yangi row noldan alohida
o‘qitiladigan dead islandga aylanmasligi kerak.

Bu attention yoki Transformer emas; typed-register dataflow saqlanadi:

```text
(a,b,op1) -> partial -> (partial,c,op2) -> final -> readout
```

## Amaldagi natija

Factorized bank 20Mda full all-pairs 96.22%, 100Mda 96.25% va 300Mda
96.40% berdi. Hidden full-grid 20M/100M/300M mos ravishda 98.66%/97.15%/
96.39% bo‘ldi. All-pairs quality scale bilan pasaymadi; hidden adaptation esa
hali monoton emas. To‘liq jadval `results/V0_37_FACTORIZED_VIRTUAL_CAPACITY.md`da.

Shu sabab 300M factorized yo‘l qabul qilindi va 500M factorized all-pairs
control bajarildi: 500M global factorized 94.72% bo‘ldi. V0.38 factor-address
router 95.55%ga ko‘tardi, V0.39 factor-pair bilinear router esa 95.43% berdi;
ikkalasi ham 300M 96.40%ni tiklamadi. Tafsilotlar
`results/V0_38_FACTOR_ADDRESS_ROUTER.md` va
`results/V0_39_FACTOR_PAIR_BILINEAR_ROUTER.md`da.

## Tajriba tartibi

1. 20M V0.36 reference bilan bir xil seed, 10k step va 64³ all-pairs grid.
2. Factorized bankning active parameter estimate va basis gradient coverage’ni
   log qilish.
3. 20M natija V0.36dan 1 punktdan ko‘p pasaymasa, 100M scale control.
4. 100M 20Mdan pastlamasa, keyin 300M va zarur bo‘lsa 500M.

## Qabul mezonlari

- 20M full all-pairs: kamida 93.95% (V0.36dan ko‘pi bilan 1 punkt past);
- active computation sparse qolishi;
- 100M full all-pairs: kamida 20M natijasiga teng yoki yuqori;
- 300M full all-pairs: 100M natijasidan pasaymasligi;
- basis/circuit gradient coverage bank kattalashganda pasaymasligi;
- hidden-stage natija full all-pairs nazoratidan alohida qayd qilinishi.

## Rad etish sharti

Factorized bank 20M, 100M va 300M all-pairs mezonlarini bajardi, ammo 500Mda
global route 94.72%ga tushdi. Route-only V0.38/V0.39 qisman yordam berdi,
lekin yetarli emas. Keyingi ish hidden composition sampler bilan birga
route-depth va 500M scale ablation bo‘ladi; V0.37 300M reference saqlanadi.

## Foydalanilgan hujjatlar

- `taklif.md` — dastlabki sparse scale g‘oyalari;
- `taklif1.md` — bajarilgan audit va rad qilingan variantlar;
- `taklif2.md` — family-local routing tarixi;
- `taklif3.md` — role-anchor va fixed role-cell natijalari;
- `taklif4.md` — shared residual tajribasi;
- `results/V0_36_MULTIPLICATIVE_REGISTER_WRITE.md` — hozirgi reference.
- `results/V0_37_FACTORIZED_VIRTUAL_CAPACITY.md` — 20M–300M scale natijalari;
- `results/V0_38_FACTOR_ADDRESS_ROUTER.md` va `results/V0_39_FACTOR_PAIR_BILINEAR_ROUTER.md`
  — 500M router ablationlari.
