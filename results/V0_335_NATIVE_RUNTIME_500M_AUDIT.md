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
| V0.330 seed17 | 1 | 500 | **39.984 ms** | 25.01 samples/s | 43 MiB | 1.000 / 4 |
| V0.330 seed17 | 128 | 100 | **126.520 ms** | 1,011.70 samples/s | 84 MiB | 2.500 / 4 |
| V0.334 seed19 | 1 | 500 | **39.912 ms** | 25.05 samples/s | 43 MiB | 1.000 / 4 |
| V0.334 seed19 | 128 | 100 | **121.450 ms** | 1,053.93 samples/s | 84 MiB | 2.500 / 4 |
| V0.330/334 mean | 1 | — | **39.948 ms** | 25.03 samples/s | — | 1.000 / 4 |
| V0.330/334 mean | 128 | — | **123.985 ms** | 1,032.38 samples/s | — | 2.500 / 4 |

Seedlar orasidagi farq batch-128da taxminan 4% bo‘ldi; bu route patterni va GPU
kernel shovqinini alohida tekshirish kerakligini ko‘rsatadi.

## Active-path hisoboti

V0.330/V0.334 checkpointlarida virtual bank `39,300` circuit va `199` factor
row deb e’lon qilingan, lekin saqlanadigan trainable model taxminan `8.99M`
parametr (`35.97 MiB`). Active estimate `1,964,480` parametr (`7.86 MiB`),
ya’ni stored modelning taxminan `21.8%`i.

Factorized digit output uchun Cartesian `2^33` klasslar to‘liq materializatsiya
qilinmaydi. Tuzatilgan analytical estimate:

| Workload | Active MAC/sample | Full 4-step MAC/sample | Active fraction | Parameter-read proxy/sample |
|---|---:|---:|---:|---:|
| Batch-1 | 1.401M | 4.688M | 29.88% | 11.81 MiB |
| Batch-128 | 3.052M | 4.695M | 65.00% | 21.47 MiB |

Bu raqamlar wall-clock kafolati emas: indexing, top-k, softmax, Python loop,
mayda CUDA launchlar va factorized gather analytical MAC hisobiga kirmaydi.

## Profiler signali

V0.334 seed19, batch-1, stats-free bir forward profilerda `Self CUDA` bo‘yicha
eng katta operatorlar quyidagicha chiqdi:

| Operator | Self CUDA |
|---|---:|
| `aten::mul` | 9.966 ms |
| `aten::as_strided` | 5.752 ms |
| `aten::addmm` | 5.623 ms |
| `aten::linear` | 4.222 ms |
| `aten::select` | 3.871 ms |
| `aten::sin` / `aten::cos` | 3.866 / 3.452 ms |
| `aten::einsum` | 2.473 ms |
| `aten::index` | 2.358 ms |

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

## V0.337 serial dispatch implementation A/B

Serial factor update uchun `einsum` o‘rniga `torch.bmm` va factor-row prefetch
opt-in yo‘llari qo‘shildi. Eski checkpoint va model matematikasi o‘zgarmadi.
V0.334 seed19da dynamic generator bilan ayni processda dispatchlar
navbatma-navbat uch raund o‘lchandi:

| Batch | `einsum` | `bmm` | `prefetch` | `prefetch_bmm` |
|---:|---:|---:|---:|---:|
| 1 | 37.932 ms | 38.274 ms (+0.90%) | 38.885 ms (+2.51%) | 39.242 ms (+3.46%) |
| 128 | 115.882 ms | 119.970 ms (+3.53%) | 116.971 ms (+0.94%) | 116.582 ms (+0.60%) |

`bmm` va prefetch variantlari output max absolute difference `0.0` bilan
numerical equivalence testidan o‘tdi, lekin paired timingda baseline’dan
sekinroq chiqdi. **V0.337 REJECTED AS A SPEED FIX**; implementation A/B va
testlar qoldi, eski `einsum` default saqlandi. To‘liq paired JSON:
`results/runs/v0_337_serial_dispatch_paired_ab.json`.

## V0.338 serial versus parallel composition control

Mavjud V0.334 checkpointida serial composition o‘rniga mavjud parallel
composition inference-only tekshirildi. Bu yangi train qilinmagan semantic
ablation; checkpoint va default config o‘zgarmadi.

| Workload | Serial | Parallel | Parallel delta | Accuracy | Digit-logit max diff |
|---|---:|---:|---:|---:|---:|
| Batch-1, all tasks | 34.613 ms | 31.497 ms | **−9.0%** | 100% / 100% | 8.4e−05 |
| Batch-128, all tasks | 107.030 ms | 78.845 ms | **−26.3%** | 100% / 100% | 0.00663 |
| Batch-4096, all tasks | 288.928 ms | 290.647 ms | +0.6% | 99.805% / 99.805% | not retained |
| Batch-4096, high-value multiply | 287.910 ms | 307.792 ms | +6.9% | 97.314% / 97.314% | 0.0536 |

Batch-1, batch-128 va high-value 4096 samplelarda prediction agreement `100%`.
Shunga qaramay parallel raw digit logitslari serial bilan aynan teng emas,
batch-size bo‘yicha speedup monotonik emas va high-value workloadda regress
qiladi. **V0.338 REJECTED FOR DEFAULT RUNTIME**, lekin parallel composition
qayta o‘qitiladigan alohida quality/runtime candidate sifatida ochiq qoldi.

To‘liq paired JSONlar: `results/runs/v0_338_serial_parallel_control_128_paired.json`,
`results/runs/v0_338_serial_parallel_control_b1_paired.json` va
`results/runs/v0_338_serial_parallel_highvalue_multiply_4096.json`.

## V0.339 parallel-continuation quality control

V0.334 seed19 checkpointi parallel composition bilan 1,000 qadam davom ettirildi.
Bu tajriba parallel dispatchning o‘zini tezlashtirish emas, parallel rejimda
qayta moslashtirilgan weightlar hard composition sifatini yaxshilaydimi degan
savolga javob beradi. Keyin V0.334 baseline va V0.339 checkpoint bir xil
seedlangan `d=4`, `multiply`, `80..95` batchda serial/parallel inference bilan
paired o‘lchandi (`4096` sample, `3` round, `3` iteration).

| Checkpoint | Inference mode | Accuracy | Mean latency / batch |
|---|---|---:|---:|
| V0.334 seed19 | serial | 90.161% | 451.383 ms |
| V0.334 seed19 | parallel | 90.161% | 466.611 ms |
| V0.339, 1k-step parallel continuation | serial | **91.528%** | 449.176 ms |
| V0.339, 1k-step parallel continuation | parallel | **91.528%** | 452.146 ms |

Hard fixed-depth slice `+1.367` percentage points yaxshilandi, prediction
agreement har checkpointda `100%` bo‘ldi. Ammo bu natija umumiy evalga hali
ko‘chmadi: V0.339 1k-step eval accuracy `99.414%`, V0.334 esa taxminan
`99.512%` edi. Parallel mode ham ikkala checkpointda serialdan tezroq emas
(V0.334da `+3.37%`, V0.339da `+0.66%`). Shuning uchun V0.339 **defaultga
promote qilinmadi**; u hard multiply uchun qayta trening signali sifatida
saqlandi, lekin parallel composition runtime yechimi deb qabul qilinmadi.

To‘liq JSONlar: `results/runs/v0_338_baseline_highvalue_d4_4096.json` va
`results/runs/v0_339_parallel_candidate_highvalue_d4_4096.json`.

## V0.340 dynamic torch.compile smoke

Dynamic-register serving wrapperi factorized digit logitsni qaytaradigan
compact output bilan `torch.compile(mode="reduce-overhead")` orqali tekshirildi.
V0.334 seed19, serial mode, batch-1 eager accuracy `100%` bo‘ldi. Inductor
kompilyatsiyasi `15.84 s`dan keyin `BackendCompilerFailed` bilan tugadi:
`Cannot find a working triton installation`. Shuning uchun compiled latency
va numerical-equivalence raqamlari mavjud emas; fallback bilan o‘lchash
compiled speedupni isbotlamaydi.

**V0.340 TOOLCHAIN-BLOCKED, DEFAULT O‘ZGARMADI.** Keyingi runtime tajribasi
Triton/MSVC mos build muhiti yoki modelning dynamic route/gather yo‘lini
qamrab oladigan native CUDA kernel talab qiladi.

## Qaror

**V0.335: runtime muammosi tasdiqlandi, quality default o‘zgarmadi.**

500M model sifatda yetakchi bo‘lsa ham, hozirgi PyTorch dispatch yo‘li
batch-1da taxminan `40 ms` turadi. Peak VRAM atigi `43 MiB`, shuning uchun
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
