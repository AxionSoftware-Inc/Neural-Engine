# Runtime — Qwen fixed-shape CUDA Graph replay

**Sana:** 2026-09-09  
**Branch:** `exp/track-runtime`  
**Qaror:** `POSITIVE OPT-IN`; trained-serving audit and shape-cache integration
remain open.

## Maqsad

Qwen one-token sparse pathida router qarori va selected-group FFN matematikasi
to‘g‘ri bo‘lsa ham, eager PyTorch ko‘p kichik CUDA launchlarni bajaradi. Shu
fixed-shape yo‘lni CUDA Graph ichida capture qilib, keyingi decode qadamlarida
faqat `replay()` bilan ishlatish launch/Python overheadini kamaytirishi mumkin.

Bu test router, circuit bank, active budget yoki output formulasini
o‘zgartirmaydi. `batch=1, sequence=1`, Qwen3-0.6B, float32, RTX 3060 va sakkiz
almashtirilgan qatlam `[0,4,8,12,16,20,24,26]` ishlatildi. Child’lar runtime
smoke uchun yaratildi; router o‘qitilmagan, shuning uchun bu hujjat yangi
quality natijasi emas.

## Natijalar

| Active groups | Iterations | Dense parent | Sparse eager fast | CUDA Graph replay | Graph / parent | Graph / eager | Max graph-eager logit error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| K=6/8 | 50 | `29.056 ms` | `33.963 ms` | `17.087 ms` | `0.588x` | `0.503x` | `1.24e-5` |
| K=5/8 | 100 | `27.843 ms` | `30.552 ms` | `14.754 ms` | `0.530x` | `0.483x` | `1.57e-5` |
| K=5/8 | 200 | `29.371 ms` | `31.919 ms` | `13.730 ms` | `0.467x` | `0.430x` | `1.57e-5` |

K=5 bizning sifat bo‘yicha tasdiqlangan past-budget operating point’imiz:
oldingi ikki-seed quality auditida CE deltalari `+0.03881/+0.04036` bo‘lib,
`+0.05` gate’dan o‘tgan. Bu benchmark shu trained checkpointni qayta
baholamaydi; u K=5 routing/dispatch shakli bilan CUDA Graph overheadini
tekshiradi.

## Input-buffer update parity

Capture’dan keyin boshqa token ID GPU’dagi captured input buffer’ga `copy_`
qilindi va graph replay natijasi shu yangi tokenning eager forward’i bilan
solishtirildi. Max logit farqi `1.05e-5`, mean farq `1.33e-6` bo‘ldi. Demak
fixed-shape graph faqat capture paytidagi bitta tokenni qaytarmayapti; input
storage’ni update qilib yangi tokenni qayta hisoblayapti.

Bir qatlamli dastlabki screen ham graph foydasini ko‘rsatdi: parent
`31.045 ms`, eager `30.448 ms`, graph `10.013 ms`, graph/eager `0.329x`.
Asosiy qaror sakkiz qatlamli natijalarga tayangan.

## Talqin

Bu avvalgi eager natijalar bilan muhim farq qiladi: K=5 sparse yo‘l eager
holatda dense parentdan `1.087x–1.097x` sekin edi, graph replay esa shu
smokeda dense parentning `0.467x–0.530x` vaqtini oldi. Demak one-token
muammosining katta qismi routing/circuit matematikasidan emas, fixed-shape
launch overheadidan kelishi mumkin.

Graph va eager outputlari bir xil input/modelda logit bo‘yicha `1.6e-5`
ichida qoldi. Bu inference reduction/timing farqiga mos parity signalidir;
quality gate yoki teacher CE o‘rnini bosmaydi. Graph replay statik shape va
statik model parametrlarini talab qiladi: yangi tokenlar uchun input buffer
copy qilinishi, KV-cache/use-cache yo‘li va dynamic batch/sequence alohida
tekshirilishi kerak.

## Qaror va keyingi qadam

- Fixed-shape CUDA Graph replay’ni runtime uchun `ACCEPTED OPT-IN` deb qabul
  qildim; bu katta runtime signali.
- Input-buffer update fixed-shape, `use_cache=False` smoke’da parity bilan
  tasdiqlandi. Default serving yo‘liga hali qo‘shilmadi: trained K=5/K=6
  model, `use_cache` bilan haqiqiy decode va bir nechta shape audit qilinishi
  kerak. Generic Transformers `StaticCache` graph capture esa alohida
  screen’da parity bermadi va qabul qilinmadi; buning uchun custom static KV
  tensor yo‘li kerak bo‘ladi.
- Keyingi ish graph shape-cache/proper input-copy wrapper va trained K=5
  quality-parity benchmarki. Graph modelni o‘zgartirmaydi, shu sabab Qwen K=4
  router muammosini hal qilgan deb talqin qilinmaydi.
- Fused extension smoke build/synchronizationda bir necha daqiqa progresssiz
  qoldi va to‘xtatildi; undan hech qanday performance claim chiqarilmadi.

## Reproduksiya

```powershell
python -u benchmark_qwen_single_token_cuda_graph.py `
  --layers 0,4,8,12,16,20,24,26 --active-experts 5 `
  --iterations 200 --warmup 50
```

## Artifact

- `benchmark_qwen_single_token_cuda_graph.py`
- `benchmark_qwen_static_cache_graph.py` — rejected `StaticCache` diagnostic
- Existing trained quality reference: `results/V0_193_QWEN_CORRECTION_DISPATCH_AUDIT.md`

## `use_cache=True` diagnostic

Generic Transformers `StaticCache` bilan graph capture’ni parent-only nazoratda
ham tekshirdim. `batch=1`, prefix length 4, max cache length 32, 5 replay
iterations smoke’da capture-vs-eager max logit xatosi `12.78`, replay-vs-eager
`1.74`, alternate-input replay-vs-eager `7.60` bo‘ldi. Bu parity emas, shuning
uchun bu yo‘lning latency raqamlari ishlatilmaydi. Muammo sparse child’ga xos
emas: parent-only nazorat ham yiqildi. Current Qwen/Transformers cache
`index_copy_`/internal cache state’i generic CUDA Graph capture uchun xavfsiz
emas; keyingi serving integratsiyasi explicit fixed KV buffers va custom
cache-update kernelini talab qiladi.
