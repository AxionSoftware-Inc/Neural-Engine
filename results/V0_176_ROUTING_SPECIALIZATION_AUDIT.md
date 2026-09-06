# V0.176 — Routing specialization audit

Bu bosqich V0.175 dagi asosiy signalni — controlled task allocation learned
routingdan yaxshiroq ekanini — arxitektura orqali qayta olishga urindi. Maqsad
task ID'ni to‘g‘ridan-to‘g‘ri circuitga hard-map qilish emas, balki semantic
family signalini yoki training gradientini routingga berish edi.

## 1. StableFamilyRouter: fizik family partition

`NeuralEngineV0` ga `router_variant=family_local` qo‘shildi. Router bankni
shared fallback va family-local bloklarga ajratadi; 8/16/32 bank screen bir xil
`active_circuits=2`, `d_model=128`, `rank=8` benchmarkda o‘tkazildi.

Ikki family (primitive vs composition), 5000 qadam, seed 17:

| Bank | In-domain | Held-out | Dead circuits |
|---:|---:|---:|---:|
| 8 | 63.36% | 48.49% | 0.0% |
| 16 | 63.07% | 46.90% | 0.0% |
| 32 | 61.93% | 44.43% | 0.0% |

Natija: 8-bankda global learned `46.64%`dan yuqori bo‘ldi, lekin capacity
oshganda held-out pasaydi. **Rad qilindi:** fizik partition bank kattalashganda
har bir family ichidagi route generalizationni saqlamadi.

To‘rt semantik family (arithmetic, comparison/parity, order-statistics,
composition) 1000-qadam pilotida ham signal bermadi:

| Bank | Families | Held-out | Dead circuits |
|---:|---:|---:|---:|
| 8 | 2 | 25.39% | 0.0% |
| 16 | 4 | 26.33% | 18.75% |
| 32 | 4 | 25.96% | 6.25% |

Bu alohida qat’iy rejim davom ettirilmadi.

## 2. Family-conditioned global router

`router_variant=family_conditioned` global bankni saqlaydi va family embeddingni
faqat router query'iga qo‘shadi. 1000-qadam pilot, seed 17:

| Bank | In-domain | Held-out |
|---:|---:|---:|
| 8 | 39.14% | 25.70% |
| 16 | 40.26% | 28.10% |
| 32 | 36.98% | 26.30% |

Global learned pilot bilan solishtirganda (`27.14 / 26.33 / 25.86%` held-out)
barqaror ustunlik yo‘q. **To‘liq screen’ga kengaytirilmadi.**

## 3. Training-only soft routing

Hard top-k routerga gradient faqat tanlangan circuitlardan boradi degan gipoteza
tekshirildi. `soft_routing_temperature=0.5` training paytida candidate pooldagi
barcha rowsni aralashtiradi; eval yana hard top-2 ga qaytadi. Bu inference
active pathni o‘zgartirmaydi, lekin training VRAM taxminan `36 → 59 MB` bo‘ldi.

1000-qadam, seed 17, doimiy soft route:

| Bank | In-domain | Held-out |
|---:|---:|---:|
| 8 | 36.69% | 26.77% |
| 16 | 38.31% | 25.36% |
| 32 | 37.45% | 25.65% |

Softni dastlabki 500 qadam bilan cheklab, keyin hard top-k ga o‘tish ham:

| Bank | In-domain | Held-out |
|---:|---:|---:|
| 8 | 37.06% | 26.77% |
| 16 | 39.58% | 26.20% |
| 32 | 37.71% | 25.34% |

**Rad qilindi:** bu benchmark va hozirgi circuit parametrlarida soft training
hard inference mismatchini foydali signalga aylantirmadi.

## Qaror va keyingi yo‘l

V0.175 dagi ikki-seed controlled allocation hozirgi eng yaxshi diagnostic:
32-bank held-out o‘rtachasi `51.86%`, global learned esa `47.18%`. V0.176
variantlari bu gapni yopmadi. Demak keyingi ish yana family partition yoki
soft mixture qo‘shish emas.

Keyingi arxitektura gipotezasi — **routerga task lossdan tashqari aniq route
learning signal berish**: training paytida controlled route targetlari bilan
router key/tree score'larini auxiliary loss orqali o‘qitish, keyin inference'da
hard learned route ishlatish. Bu controlled natijani ko‘chirishga urinadi,
lekin final executionni fixed mappingga qamab qo‘ymaydi. Avval 8/16/32 pilot,
keyin faqat ikki seedda takrorlansa 5000-qadam screen qilinadi.
