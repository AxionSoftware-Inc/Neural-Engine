# V0.335 — Native dynamic-register 500M runtime audit

**Sana:** 2026-09-13  
**Branch:** `exp/track-native-engine`  
**GPU:** NVIDIA GeForce RTX 3060  
**Maqsad:** V0.330/V0.334 sifat bo‘yicha yetakchi 500M virtual-capacity
checkpointlarining serving yo‘lidagi haqiqiy latency, throughput va active-path
xarajatini o‘lchash.

## Protokol

- Checkpointlar va model og‘irliklari o‘zgartirilmadi.
- CUDA, `matmul_precision=highest`, balanced batch, 3 warm-up iteration.
- Serving yo‘li `collect_state_stats=False` ekvivalenti bilan ishladi.
- Batch-1 uchun 500, batch-128 uchun 100 o‘lchangan iteration.
- V0.330 seed17 va V0.334 seed19 bir xil dinamik-register evaluator bilan
  o‘lchandi.
- Dynamic checkpointlarda `benchmark.py` ilgari `NeuralEngineV0` factory’ga
  tushib qolardi; V0.335da architecture-aware loader qo‘shildi.

## Natija

| Checkpoint | Batch | Iterations | Latency / batch | Throughput | Peak VRAM | Avg executed steps |
|---|---:|---:|---:|---:|---:|---:|
| V0.330 seed17 | 1 | 500 | **59.072 ms** | 16.93 samples/s | 43 MiB | 2.000 / 4 |
| V0.330 seed17 | 128 | 100 | **84.342 ms** | 1,517.64 samples/s | 85 MiB | 2.828 / 4 |
| V0.334 seed19 | 1 | 500 | **57.702 ms** | 17.33 samples/s | 43 MiB | 2.000 / 4 |
| V0.334 seed19 | 128 | 100 | **92.765 ms** | 1,379.83 samples/s | 85 MiB | 2.828 / 4 |
| V0.330/334 mean | 1 | — | **58.387 ms** | 17.13 samples/s | — | 2.000 / 4 |
| V0.330/334 mean | 128 | — | **88.553 ms** | 1,448.73 samples/s | — | 2.828 / 4 |

Seedlar orasidagi farq batch-128da taxminan 10% bo‘ldi; bu sifat farqidan ko‘ra
router route patterni va GPU kernel shovqinini alohida tekshirish kerakligini
ko‘rsatadi.

## Active-path hisoboti

V0.330/V0.334 checkpointlarida virtual bank `39,300` circuit va `199` factor
row deb e’lon qilingan, lekin saqlanadigan trainable model taxminan `8.99M`
parametr (`35.97 MiB`). Active estimate `1,964,480` parametr (`7.86 MiB`),
ya’ni stored modelning taxminan `21.8%`i.

Factorized digit output uchun Cartesian `2^33` klasslar to‘liq materializatsiya
qilinmaydi. Tuzatilgan analytical estimate:

| Workload | Active MAC/sample | Full 4-step MAC/sample | Active fraction | Parameter-read proxy/sample |
|---|---:|---:|---:|---:|
| Batch-1 | 2.496M | 4.688M | 53.25% | 18.25 MiB |
| Batch-128 | 3.408M | 4.692M | 72.63% | 23.58 MiB |

Bu raqamlar wall-clock kafolati emas: indexing, top-k, softmax, Python loop,
mayda CUDA launchlar va factorized gather analytical MAC hisobiga kirmaydi.

## Profiler signali

V0.334 seed19, batch-1, stats-free bir forward profilerda `Self CUDA` bo‘yicha
eng katta operatorlar quyidagicha chiqdi:

| Operator | Self CUDA |
|---|---:|
| `aten::mul` | 12.756 ms |
| `aten::as_strided` | 10.090 ms |
| `aten::addmm` | 7.052 ms |
| `aten::select` | 5.860 ms |
| `aten::index` | 5.272 ms |
| `aten::linear` | 5.206 ms |
| `aten::einsum` | 4.775 ms |
| `aten::cos` / `aten::sin` | 4.616 / 3.864 ms |

Profiler launch overhead sabab bu summalar benchmark latency bilan bir xil
emas, ammo signal aniq: bottleneck bitta katta GEMM emas, balki ko‘p mayda
elementwise/view/index operatorlari va factorized route dispatch yig‘indisi.
Shuning uchun keyingi fused prototip avval `mul` + view/index yo‘li va route
gatherni kamaytirishi, keyin `einsum`/linearlarni packed batchga birlashtirishi
kerak.

Profiler reproduksiyasi:

```powershell
python profile_native_runtime.py --checkpoint results/checkpoints/v0_334_500m_nonmod_targeted_highvalue_multiply25_stable_factor_growth_seed19_4000.pt --batch-size 1 --warmup 5 --row-limit 35 --no-stats
```

## Qaror

**V0.335: runtime muammosi tasdiqlandi, quality default o‘zgarmadi.**

500M model sifatda yetakchi bo‘lsa ham, hozirgi PyTorch dispatch yo‘li
batch-1da taxminan `58 ms` turadi. Peak VRAM atigi `43 MiB`, shuning uchun
cheklov model sig‘imi yoki xotira yetishmasligi emas; asosiy gumon —
active circuitlarni tanlash, factorized gather va recurrent step ichidagi
ko‘p kichik kernel/dispatchlar.

Bu natija Native Engine g‘oyasini rad qilmaydi: active parameter va analytical
MAC qisqarishi real. Lekin “kam active parametr avtomatik ravishda tezroq
ishlaydi” degan da’vo hozirgi unfused implementation uchun tasdiqlanmadi.

Keyingi ish quality tuning emas, shu modelning numerical-equivalent profiler
auditi va fixed-shape/fused dispatch prototipi. Compiled/fused variant faqat
latency va output equivalence birga o‘lchangandan keyin defaultga nomzod
bo‘ladi.

## Reproduksiya

```powershell
python benchmark.py --checkpoint results/checkpoints/v0_330_500m_nonmod_targeted_highvalue_multiply25_stable_factor_growth_seed17_4000.pt --device cuda --batch-size 1 --iterations 500 --balanced-batch --no-stats
python benchmark.py --checkpoint results/checkpoints/v0_330_500m_nonmod_targeted_highvalue_multiply25_stable_factor_growth_seed17_4000.pt --device cuda --batch-size 128 --iterations 100 --balanced-batch --no-stats
python benchmark.py --checkpoint results/checkpoints/v0_334_500m_nonmod_targeted_highvalue_multiply25_stable_factor_growth_seed19_4000.pt --device cuda --batch-size 1 --iterations 500 --balanced-batch --no-stats
python benchmark.py --checkpoint results/checkpoints/v0_334_500m_nonmod_targeted_highvalue_multiply25_stable_factor_growth_seed19_4000.pt --device cuda --batch-size 128 --iterations 100 --balanced-batch --no-stats
```
