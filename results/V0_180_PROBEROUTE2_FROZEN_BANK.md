# V0.180 — ProbeRoute-2 frozen-bank sinovi

## Maqsad

Ekspert taklif qilgan ikki bosqichli router tekshirildi:

1. retriever butun 32-circuit bankdan `M=8` candidate pool oladi;
2. pair-selector shu pool ichidagi `C(8,2)=28` juftlikni individual utility va
   rank-8 pair interaction bilan baholaydi;
3. inference'da faqat tanlangan 2 circuit ishlaydi;
4. training'da bitta recurrent step uchun qo‘shimcha alternative pair probe
   qilinadi va final CE farqi routerga signal bo‘ladi.

Bu bosqichda circuit body va correction vaznlari muzlatildi. Faqat yangi
router parametrlari train qilindi. `200` imitation qadamidan keyin `1800`
probe-cost qadam bajarildi. Har bir tajriba 8 ta balanced training batch,
2 ta held-out batch (`240` misol) va bir xil deterministik ProbeRoute
initialization bilan qayta o‘lchandi.

Kod: `neural_engine/router.py` dagi `ProbeRouteRouter`, model integratsiyasi
`neural_engine/model.py`, trainer `probe_route_frozen.py`.

## Natijalar

Jadvaldagi `Δ CE` va `Δ acc` ProbeRoute natijasi minus o‘sha checkpointdagi
frozen hierarchical teacher natijasidir. CE uchun manfiy yaxshi; accuracy uchun
musbat yaxshi.

| Rejim | Seed | Teacher acc | ProbeRoute acc | Δ CE | Δ acc | Dead circuits |
|---|---:|---:|---:|---:|---:|---:|
| full | 17 | 51.25% | 50.42% | +0.0105 | −0.83 pp | 0/32 |
| full | 18 | 45.83% | 46.25% | −0.1810 | +0.42 pp | 1/32 |
| selection-only | 17 | 51.25% | 44.17% | +0.1181 | −7.08 pp | 0/32 |
| selection-only | 18 | 45.83% | 46.25% | −0.1076 | +0.42 pp | 0/32 |
| retrieval-only | 17 | 51.25% | 44.58% | +0.0071 | −6.67 pp | 0/32 |
| retrieval-only | 18 | 45.83% | 45.42% | −0.0021 | −0.42 pp | 0/32 |

Ikki seed o‘rtachasi:

| Rejim | O‘rtacha Δ CE | O‘rtacha Δ accuracy |
|---|---:|---:|
| full | −0.0853 | −0.21 pp |
| selection-only | +0.0053 | −3.33 pp |
| retrieval-only | +0.0025 | −3.54 pp |

Har bir 2000-step run taxminan `69–76 s` davom etdi. Dead circuit soni 0 yoki
1 bo‘ldi; demak, bu kichik sinovda collapse muammo emas.

## Tahlil

Full rejimda CE ikki seed bo‘yicha o‘rtacha yaxshilandi, lekin hard accuracy
bo‘yicha katta sakrash bo‘lmadi va ikki seedning ikkalasi ham ijobiy chiqmadi.
Selection-only va retrieval-only rejimlari alohida barqaror foyda bermadi.
Shuning uchun kuzatilgan kichik CE yaxshilanishini yangi arxitektura ustunligi
deb qabul qilib bo‘lmaydi; u joint training, seed va loss calibration'ga sezgir.

Ekspertning oldindan belgilagan gate'lari bajarilmadi:

- full rejimda o‘rtacha `+2 pp` accuracy yo‘q;
- ikkala seed ham hard accuracy bo‘yicha ijobiy emas;
- uchinchi seed (`19`) uchun frozen-bank checkpoint mavjud emas edi;
- probe training baseline'dan `<=1.5x` ekanini bu runlarda paired wall-clock
  benchmark bilan o‘lchamadik;
- 300M yoki undan katta modelga o‘tish uchun ishonchli scaling signal topilmadi.

## Muhim metodik tuzatish

Dastlabki smoke/runlarda yangi ProbeRoute parametrlari tasodifiy
initialization bilan yaratilgan edi. O‘sha raqamlar yakuniy taqqoslashdan
chiqarildi. Keyingi runlarda initialization seedga bog‘landi, shuning uchun
rejimlar objective farqini tozaroq ko‘rsatadi.

## Qaror

`ProbeRoute-2` ni asosiy default arxitektura sifatida qabul qilish yoki undan
to‘g‘ridan-to‘g‘ri 300M/700M/1B modelga o‘tish hozircha rad qilindi. Biroq
proposal butunlay befoyda deb ham yopilmadi: full rejimdagi CE signali
retriever va selectorni birgalikda, on-policy tarzda o‘qitish bo‘yicha kichik
gipotezani qoldiradi.

Keyingi ilmiy qadam capacity'ni oshirish emas, avval 32-bankda quyidagilarni
to‘g‘ri o‘lchash bo‘ladi:

1. frozen bankda candidate recall va pair-selection regretni bevosita audit
   qilish;
2. probe target'larini final corrected output bilan bir xil oracle'ga
   bog‘lash;
3. hard accuracy bilan birga p95 regret va CE'ni seed19 bilan tekshirish;
4. gate bajarilmasa, bu router objective'ni muzlatib, boshqa fundamental
   circuit allocation mexanizmini sinash.

## Fayllar

- [seed17 full JSON](runs/probe_frozen_s17_full_det.json)
- [seed18 full JSON](runs/probe_frozen_s18_full_det.json)
- [seed17 selection-only JSON](runs/probe_frozen_s17_selection_det.json)
- [seed18 selection-only JSON](runs/probe_frozen_s18_selection_det.json)
- [seed17 retrieval-only JSON](runs/probe_frozen_s17_retrieval_det.json)
- [seed18 retrieval-only JSON](runs/probe_frozen_s18_retrieval_det.json)
- [ProbeRoute-2 config](../configs/ne_capacity_probe.yaml)
