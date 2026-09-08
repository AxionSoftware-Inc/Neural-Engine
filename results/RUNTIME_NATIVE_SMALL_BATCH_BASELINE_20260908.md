# Runtime — Native Engine small-batch baseline and stats-free path

**Sana:** 2026-09-08  
**Branch:** `exp/track-runtime`  
**Checkpoint:** `results/checkpoints/ne100_gate_staged_s17_full_10000.pt`  
**GPU:** NVIDIA GeForce RTX 3060  
**Qaror:** `ACCEPTED FOR SERVING PATH`; CUDA Graph `NOT PROMOTED`, compiled
kernel hali `OPEN`.

## Maqsad

Native Engine sifati va routing objective'ini o'zgartirmasdan kichik batch va
one-sample latencydagi overheadni ajratish. Modelning odatiy `forward` yo'li
route audit uchun ko'p diagnostik tensorlar (`selected_ids`, `query_states`,
`step_logits`, `halt_logits` va boshqalar) yig'adi. Serving uchun ular kerak
emas, shuning uchun `collect_stats=False` opt-in yo'li qo'shildi.

Bu patch circuit bank, router qarori, active circuit soni yoki output
formulasini o'zgartirmaydi. Unit test bir xil inputda stats bilan va
statsiz logitsni `allclose` orqali tekshiradi.

## CUDA latency

| Model / path | Batch | Iterations | Latency / batch | Samples/s | Densega nisbatan |
|---|---:|---:|---:|---:|---:|
| Native, diagnostics | 128 | 100 | 10.871 ms | 11,775 | 0.322x |
| Native, `collect_stats=False` | 128 | 100 | 9.618 ms | 13,309 | 0.284x |
| Native, stats-free + router metadata skip | 128 | 100 | 8.077 ms | 15,848 | 0.239x |
| Dense reference | 128 | 100 | 33.802 ms | 3,787 | 1.000x |
| Native, diagnostics | 1 | 500 | 4.881 ms | 205 | 1.481x |
| Native, `collect_stats=False` | 1 | 500 | 3.737 ms | 268 | 1.134x |
| Native, stats-free + router metadata skip | 1 | 500 | 3.572 ms | 280 | 1.084x |
| Dense reference | 1 | 500 | 3.296 ms | 303 | 1.000x |

Stats-free path diagnostics-heavy baselinega nisbatan avval batch-128da
`11.5%`, batch-1da `23.5%` tezroq bo'ldi. Router metadata/entropy yig'ishni
ham stats-free yo'lda cheklaganimizdan keyingi qayta o'lchov `8.077/3.572 ms`
bo'ldi: diagnostics-heavy yo'lga nisbatan mos ravishda `25.7%/26.8%` tezroq.
Native batch-128da dense reference'dan `4.18x` tezroq, lekin batch-1da hali
dense'dan `8.4%` sekinroq. Bu sezilarli overhead kamayishi, ammo one-token
muammoni to'liq yopadigan fused kernel emas.

## Native branch qayta tekshiruvi

Runtime patchlari `exp/track-native-engine` branchiga cherry-pick qilingandan
keyin ayni checkpoint, RTX 3060, `balanced_batch` va `collect_stats=False`
protokolida qayta o‘lchandi. Batch-1 `3.304 ms` (`302.7 samples/s`), batch-128
`8.165 ms` (`15,676 samples/s`) chiqdi. Bu avvalgi runtime-branch natijalari
(`3.572/8.077 ms`) bilan bir xil tartibda; kichik farq GPU timing shovqini.
Model logitsi yoki active budget o‘zgarmadi. Shu bilan serving patchi Native
branchda ham regressiyasiz tasdiqlandi.

Native benchmarkdagi adaptive execution seed17 balanced batch uchun batch-128da
o'rtacha `1.5625`, batch-1da `1.0` internal step bo'ldi. Natijalarni shu
parametr va workload bilan qayta olish kerak; batchlar sifat benchmarki emas,
faqat runtime o'lchovidir.

## Profiling xulosasi

Diagnostic path profilerida `select`, `as_strided`, `index/index_put`,
`einsum`, `nonzero` va ko'p kichik CUDA launchlar ko'rindi. Stats-free path
`index_put`/diagnostic allocationlarni kamaytiradi, lekin router va circuit
bankning mayda dispatchlari saqlanadi. `torch.compile(reduce-overhead)` smoke
testi bu muhitda Triton o'rnatilmagani uchun compiler backend bosqichida
ishlamadi; bu model sifati yoki arxitektura rad javobi emas, muhit cheklovi.

## Reproduksiya

```powershell
python benchmark.py --checkpoint results/checkpoints/ne100_gate_staged_s17_full_10000.pt --device cuda --batch-size 128 --iterations 100 --balanced-batch --no-stats
python benchmark.py --checkpoint results/checkpoints/ne100_gate_staged_s17_full_10000.pt --device cuda --batch-size 1 --iterations 500 --balanced-batch --no-stats
python profile_native_runtime.py --checkpoint results/checkpoints/ne100_gate_staged_s17_full_10000.pt --batch-size 1 --no-stats
```

## Static CUDA Graph smoke

Adaptive halting dynamic bo'lgani uchun CUDA Graph'ni faqat `adaptive=False`,
stats-free, fixed-shape wrapperda tekshirdim. Logit reconstruction xatosi
`0.0` bo'ldi. Router metadata skipdan keyingi 500/200 iteration
steady-state smoke'da Graph/eager ratio batch-1da `0.959x`
(`0.651 → 0.624 ms`), batch-128da `0.985x` (`3.583 → 3.529 ms`) bo'ldi.
Bu kichik launch foydasi, lekin asosiy one-token gapni yopadigan sakrash emas;
dynamic adaptive path uchun defaultga qo'yilmadi.

Native branch qayta tekshiruvida (`exp/track-native-engine`) Graph logit xatosi
yana `0.0` bo‘ldi, ammo timing batch-1da eager `0.654 ms` va Graph `0.659 ms`
(`1.009x`), batch-128da eager `3.602 ms` va Graph `3.552 ms` (`0.986x`) chiqdi.
Demak fixed-shape Graph batch-128da faqat taxminan `1.4%` mikro-foyda beradi,
one-token muammosini yopmaydi. `benchmark_native_compile.py` qayta sinovida
esa Windows muhitida ishlaydigan Triton topilmadi; compile natijasi muhit
cheklovi sifatida `OPEN`, model yoki sifat rad javobi sifatida emas.

`torch.cuda.make_graphed_callables` uchun `router_decisions` va `soft_route`
metadata'lari GPU scalar emas, host metadata sifatida qaytarildi. Bu numerik
model outputini o'zgartirmaydi va router testlari saqlandi.

## Post-architecture-change serving recheck

2026-09-08 kuni `exp/track-native-engine` branchida opt-in register auditlari
qo‘shilgandan keyin default checkpoint yana o‘lchandi. `collect_stats=False`,
`matmul_precision=highest`, balanced batch va RTX 3060 protokoli bilan:

| Batch | Iterations | Latency / batch | Samples/s | Peak VRAM |
|---:|---:|---:|---:|---:|
| 1 | 300 | `3.195 ms` | `313` | `395 MiB` |
| 128 | 100 | `7.741 ms` | `16,535` | `468 MiB` |

Bu o‘lchovlar bridge/mixer opt-in bo‘lmaganda oldingi stats-free serving yo‘li
saqlanganini tasdiqlaydi. Quality patchlarining default runtime’ga regressiyasi
yo‘q; one-token latency muammosi esa hali compiled/fused dispatch bilan
yopilmagan.

## Float32 precision A/B

RTX 3060 TF32 yo'lini ham alohida tekshirdim. Batch-1da `highest → high`
`3.620 → 3.594 ms` (`0.993x`) bo'ldi va shu inputda logits farqi `0.0` edi.
Batch-128da esa `8.083 → 8.639 ms` (`1.069x`, ya'ni sekinroq) bo'ldi; max
logit abs error `0.007057`, mean error `0.000909`. Natija workloadga qarab
qarama-qarshi va batch-128da precision farqi ham bor, shuning uchun
`matmul_precision=high` defaultga yoki quality pathga o'tkazilmadi.

## Keyingi yo'l

`collect_stats=False` serving path sifatida saqlanadi. Keyingi runtime
eksperimenti router/circuit uchun static-shape CUDA graph yoki mavjud
Triton/CUDA toolchain bilan fused dispatchni alohida tekshirishi kerak. Bunda
quality benchmark va `collect_stats=True` audit yo'li o'zgarmasligi shart.
