# P-003/P-002 — Native Engine route-causal diagnostic

**Sana:** 2026-09-07  
**Branch:** `exp/track-native-engine`  
**Status:** `DIAGNOSTIC COMPLETE — NEXT HYPOTHESIS OPEN`

## Maqsad

100M va 300M staged checkpointlarda capacity o‘sishi bilan sifat nega
to‘yinayotganini uch bo‘lakka ajratish:

1. route fragmentation va bank coverage;
2. circuitlar funksional jihatdan collapse bo‘lyaptimi;
3. tanlangan route final outputni sababiy boshqaryaptimi.

Bu ish yangi parametr o‘qitmadi va default modelni o‘zgartirmadi.

## Protocol

Bir xil `examples_per_task=128`, evaluator seed `1712`, CUDA va checkpointning
saqlangan effective config’i ishlatildi. Route stability adaptive inference
bilan, specialization esa barcha uch recurrent step’ni majburan ishlatib
tekshirildi. Route replay ham uch fixed step causal comparison sifatida
ishlatildi.

## Route fragmentation

| Checkpoint | Bank | Used route circuits | Dead | Within-task Jaccard | Between-task union Jaccard |
|---|---:|---:|---:|---:|---:|
| 100M seed17 | 7,552 | 6,305 | 16.51% | 0.0091 | 0.0855 |
| 100M seed18 | 7,552 | 6,259 | 17.12% | 0.0104 | 0.0819 |
| 300M seed17 | 22,800 | 8,446 | 62.96% | 0.0129 | 0.0393 |
| 300M seed18 | 22,800 | 8,252 | 63.81% | 0.0182 | 0.0422 |

300M bank uch baravar katta bo‘lsa ham, used fraction 100M’dagi taxminan
83%dan 36%gacha tushdi. Tasklararo route overlap ham kamaydi. Bu yangi
capacity active computationga muntazam aylanmayotganini ko‘rsatadi.

## Functional collapse tekshiruvi

| Checkpoint | Selected bank fraction | Selected pair cosine | Route/candidate norm ratio | Used-circuit task entropy |
|---|---:|---:|---:|---:|
| 100M seed17 | 94.12% | 0.0904 | 0.6525 | 1.084 |
| 100M seed18 | 94.36% | 0.0926 | 0.6618 | 1.060 |
| 300M seed17 | 50.59% | 0.1034 | 0.7488 | 0.499 |
| 300M seed18 | 48.90% | 0.1016 | 0.7245 | 0.533 |

300M circuitlar bir xil outputga to‘liq collapse bo‘lmagan: selected pair
cosine faqat biroz yuqori. Ammo task entropy keskin tushgan. Bu “circuitlar
umuman o‘rganmadi”dan ko‘ra, ularning route orqali qayta ishlatilishi va
tasklararo coverage’i buzilganini ko‘rsatadi.

## Causal route replay

| Checkpoint | Natural accuracy | Mean absolute swapped-route Δaccuracy | Mean swapped-route loss change |
|---|---:|---:|---:|
| 100M seed17 | 85.811% | 0.082 pp | +0.00042 |
| 100M seed18 | 84.938% | 0.216 pp | −0.00116 |
| 300M seed17 | 86.082% | 0.086 pp | +0.00201 |
| 300M seed18 | 84.573% | 0.056 pp | −0.00121 |

Route almashtirilganda logitlarda o‘rtacha `~0.14–0.15` absolute delta bor,
lekin hard accuracy o‘zgarishi juda kichik va seedlar bo‘yicha bir xil emas.
Demak router yo‘l tanlayapti, ammo recurrent state/shared input yo‘li circuit
farqini final qarorga yetarlicha olib bormayapti.

## Qaror

Muammo **faqat candidate router score’i emas**. Hozirgi eng kuchli sabablar:

- bank kengayganda route fragmentation;
- yangi circuitlarning foydali route’ga kirish/exposure muammosi;
- circuit correction’ning recurrent state va shared input tomonidan bosilishi.

Yangi 500M/700M/1B training hozircha boshlanmaydi.

## Keyingi hypothesis

`H1 — prefix-preserving expansion`: bank kengayganda inherited router pathlari
saqlanadi, yangi child-level route geometriyasi qisqa transition window’da
attenuate/freeze qilinadi va keyin asta-sekin ochiladi. Active budget majburan
oshirilmaydi; circuit body va output head o‘zgarmaydi.

Bu hypothesis route fragmentation sababmi yoki faqat qo‘shimcha training
foydasimi, shuni ajratish uchun 100M va 300M’da frozen staged controls bilan
solishtiriladi. Gate: held-out hard accuracy, dead fraction, task entropy,
route-swap sensitivity va active cost birgalikda.

## H1 100M pilot

H1 avval 100M staged clamp checkpointdan ikki seedda 2,000 continuation qadam
bilan tekshirildi. Treatment dastlabki 1,000 qadamda `routing_capacity=1408`,
`routing_depth=4`ni saqlab, 1,001-qadamda full `7552/depth5`ga o‘tdi. Control
shu checkpointdan full route bilan aynan 2,000 qadam yurdi.

| Seed | H1 accuracy | Control accuracy | H1 − control | H1 dead | Control dead |
|---:|---:|---:|---:|---:|---:|
| 17 | 80.42% | 80.86% | −0.44 pp | 5.36% | 4.85% |
| 18 | 79.51% | 79.64% | −0.13 pp | 5.89% | 5.68% |

Route overlap/task-entropy ayrim seedda ozgina yaxshilangan bo‘lsa ham, quality
gate’dan o‘tmadi. Pilotda H1 controlga qaraganda 1,000 ta kam full-route qadam
olganini hisobga olib, bu natija “prefix-preserving expansion imkonsiz” degan
qat’iy hukm emas. Biroq hozircha 300M/full-budget continuationga arzimaydigan
signal berdi va `NOT PROMOTED` qilindi.

Keyingi Native sinov route’ni shunchaki saqlash emas, circuit output’ning
recurrent state va final target bilan causal bog‘lanishini kuchaytiradigan
minimal opt-in interface/credit mexanizmi bo‘ladi. Avval 20M tez pilot, keyin
ikki seedli 100M nazorat.

## Reproduction artifacts

- `results/runs/route_diag_ne100_s17.json`
- `results/runs/route_diag_ne100_s18.json`
- `results/runs/route_diag_ne300_s17.json`
- `results/runs/route_diag_ne300_s18.json`
- `results/runs/specialization_diag_ne100_s17.json`
- `results/runs/specialization_diag_ne100_s18.json`
- `results/runs/specialization_diag_ne300_s17.json`
- `results/runs/specialization_diag_ne300_s18.json`
- `results/runs/replay_diag_ne100_s17.json`
- `results/runs/replay_diag_ne100_s18.json`
- `results/runs/replay_diag_ne300_s17.json`
- `results/runs/replay_diag_ne300_s18.json`
