# Taklif 4 — shared residual/basis circuit bank

Status: **sinab ko‘rildi; rad qilindi**

## Muammo

V0.28–V0.34da route o‘zgarishlari 20Mda ba’zan foyda berdi, ammo 100Mga
barqaror ko‘chmadi. Hozir har bir circuit row mustaqil `down/up/bias` tensoridir.
Katta bankda row’larning ko‘pi kam gradient oladi va useful computationga
aylanmaydi.

Bu taklif routingni yana qattiq bo‘lishdan oldin circuit bankning o‘ziga umumiy
hisoblash yo‘lini qo‘shadi. `SharedResidualMicroCircuitBank` implement qilindi
va 20M nazoratli run’da tekshirildi.

## Yangi circuit graph

```text
query
  ├── shared low-rank transform ─────────────┐
  └── routed circuit adapters (top-k rows) ──┴── register write
```

Shared transform har stage’da bir marta ishlaydi. Selected circuit rows esa
faqat value-dependent residual/adapterni hisoblaydi. Shuning uchun route
adashsa ham umumiy primitive computation butunlay yo‘qolmaydi.

## Arxitektura

`SharedResidualMicroCircuitBank` quyidagilarga ega bo‘ladi:

- umumiy `shared_down/shared_up/shared_bias` low-rank transform;
- mavjud per-circuit `down/up/bias` residual rows;
- serial mode’da shared residual avval, routed residual keyin ishlaydi;
- LazyAdamW shared qatlamni doimiy, circuit rowsni esa faqat tanlangan IDs
  bo‘yicha yangilaydi.

Bu attention yoki Transformer emas. Typed-register dataflow o‘zgarishsiz qoladi.

## Nega bu avvalgi variantlardan boshqa

- V0.32 bankni qattiq familylarga bo‘ldi — local width kamaydi.
- V0.33 coarse learned anchor’da collapse bo‘ldi.
- V0.34 fixed role cell’da 100M local routing underfit qildi.
- V0.35 global router saqlanadi, lekin barcha route’lar shared primitive path
  bilan umumiy gradient oladi.

## Amaldagi natija

20M all-pairs full-domain accuracy 57.61%, hidden-stage accuracy 60.49% bo‘ldi.
Reference 58.10% va 64.89% edi. Shared path umumiy gradient oldi, ammo
independent residual rows credit-assignment muammosini hal qilmadi.

Shuning uchun shared residual standalone yechim sifatida rad qilindi. Tafsilot
`results/V0_35_SHARED_RESIDUAL_BANK.md`da.

## Sinov tartibi

1. 20M global-router reference va shared-residual variantni bir xil seed,
   steps, batch, loss va `64³` grid bilan taqqoslash.
2. Hidden-stage 5,000 qadamdan keyin full hidden gridni o‘lchash.
3. 20M baseline saqlansa, ayni config bilan 100Mni train qilish.
4. 100M reference tiklansa yoki oshsa, keyin 300Mga o‘tish. Bu mezon bajarilmadi.

## Qabul qilish mezonlari

- 20M all-pairs `58.10%`dan sezilarli pastlamasligi;
- 20M hidden `64.89%`dan yuqori yoki kamida teng bo‘lishi;
- 100M all-pairs `59.23%` va hidden `89.16%` reference’lariga yaqinlashishi;
- active path sparse qolishi;
- shared path active estimate’ni haddan tashqari oshirmasligi;
- circuit usage va gradient coverage model kattalashganda yaxshilanishi.

## Falsifikatsiya

Agar shared residual 20Mda ham baseline’ni tiklamasa, muammo mustaqil circuit
rowsdagina emas. Keyingi yo‘l factorized/reusable basis capacity va
scale-invariant addressni sinash bo‘ladi; yangi reja `taklif5.md`da.

## Bog‘liqlik

Bu taklif `taklif.md`ning 3, 10, 13, 15, 16, 22 va 23-bo‘limlari, V0.32,
V0.33 va V0.34 natijalariga asoslangan.
