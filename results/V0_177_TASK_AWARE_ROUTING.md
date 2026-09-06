# V0.177 — Task-aware routing and route-target audit

## Maqsad

V0.175 controlled allocation learned routingdan yaxshiroq ekanini ko‘rsatdi.
V0.176 family partition, family-conditioned query va training-only soft routing
bu farqni yopmadi. Shu bosqichda ikki savol alohida tekshirildi:

1. Routerga task identity berilsa, u controlled routingdagi foydali signalni
   hard pathni majburlamasdan o‘rganadimi?
2. Task uchun oldindan belgilangan circuit group'ni auxiliary route-target loss
   bilan o‘rgatish foyda beradimi?

Task-context faqat router query'iga qo‘shildi (`task_context_update=false`).
Inference baribir learned hard top-k route ishlatadi; circuit ID tashqaridan
majbur qilinmadi.

## Route-target auxiliary pilot

`route_target_supervision` task ID'ni contiguous circuit group targetiga aylantirib,
hierarchical tree path va circuit key score uchun yordamchi loss beradi. Bu
controlled route'ni training signal sifatida ko‘chirishga urinish edi.

Seed 17, 1000 qadam, 32 circuit, active 2:

| Variant | Held-out | In-domain | Router entropy |
|---|---:|---:|---:|
| Global baseline | 27.14% | 37.86% | — |
| Target weight 0.01 | 27.03% | 36.67% | 0.777 |
| Target weight 0.03 | 26.82% | 36.09% | 0.477 |
| Target weight 0.10 | 26.12% | 35.49% | 0.190 |

Task-context bilan `weight=0.03` ham 26.28%, 26.46% va 26.33% held-out
(8/16/32 bank) berdi. Auxiliary signal router entropy'ni juda pasaytirdi,
lekin final hard selection sifatini oshirmadi.

**Qaror: rad qilindi.** Bu target mapping haqiqiy final corrected output
cost'idan olinmagan va modelni arbitrary group assignmentga yopishtiradi.

## Task-context capacity screen

Bir xil config, `d_model=128`, `rank=8`, `active_circuits=2`, 5000 qadam,
seeds 17 va 18. Global learned baseline va V0.175 controlled allocation bilan
solishtirildi.

| Bank | Global learned | Task-context | Controlled task |
|---:|---:|---:|---:|
| 8 | 47.83% | **51.94%** | 49.74% |
| 16 | 45.75% | 46.00% | 49.03% |
| 32 | 47.18% | **48.62%** | **51.86%** |

Seed-level task-context held-out values:

| Bank | Seed 17 | Seed 18 |
|---:|---:|---:|
| 8 | 51.43% | 52.45% |
| 16 | 43.12% | 48.88% |
| 32 | 46.38% | 50.86% |

Task-context 32-bankda global baselinega nisbatan o‘rtacha `+1.45` punkt
berdi. Lekin capacity oshishi monoton emas: 8-bankdagi kuchli natija 16-bankda
yo‘qoladi va 32-bankda faqat qisman qaytadi. Shu sabab bu yechim routing
signalining yetishmasligini ko‘rsatadi, ammo fundamental capacity muammosini
hal qilmaydi.

## Texnik o‘zgarishlar

- `NeuralEngineV0` task-aware query uchun mavjud `task_context` yo‘lidan
  foydalanadi; yangi config: `configs/ne_capacity_task_context.yaml`.
- `HierarchicalRouter` training-only soft candidate mixture va route-target
  auxiliary diagnostikasini qo‘llab-quvvatlaydi.
- `family_local` va `family_conditioned` variantlari V0.176 auditda saqlandi.
- Har ikkala yangi mexanizm hard inference active pathni saqlaydi.

## Xulosa va keyingi qadam

Bu bosqich 300M/700M/1B ga o‘tishni oqlamaydi. Eng kuchli va takrorlangan
signal hali ham controlled task allocation: bank kattalashganda sifatni
saqlab qoladi va 32-bankda learned baseline'dan 4.68 punkt yuqori.

Task-context optional baseline sifatida saqlanadi, route-target esa default
emas. Keyingi arxitektura sinovi task ID'ni ko‘rsatish emas, balki final
corrected output cost'ini bevosita o‘rganuvchi cost-aware router bo‘lishi kerak;
uning hard-selection regret'i held-outda o‘lchanadi. Agar u controlled gapni
yopmasa, bankni kattalashtirish emas, circuit specialization/interface'ini
qayta loyihalash kerak bo‘ladi.
