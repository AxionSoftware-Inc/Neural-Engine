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
| Dense reference | 128 | 100 | 33.802 ms | 3,787 | 1.000x |
| Native, diagnostics | 1 | 500 | 4.881 ms | 205 | 1.481x |
| Native, `collect_stats=False` | 1 | 500 | 3.737 ms | 268 | 1.134x |
| Dense reference | 1 | 500 | 3.296 ms | 303 | 1.000x |

Stats-free path diagnostics-heavy baselinega nisbatan batch-128da `11.5%`,
batch-1da `23.5%` tezroq bo'ldi. Native batch-128da dense reference'dan
`3.51x` tezroq, lekin batch-1da hali dense'dan `13.4%` sekinroq. Shuning
uchun bu patch overheadni kamaytirdi, ammo one-token muammoni to'liq hal
qilmadi.

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
`0.0` bo'ldi. 100/200 iteration steady-state smoke'da Graph/eager ratio
batch-1da `0.968x` (`0.761 → 0.737 ms`), batch-128da `0.989x`
(`3.687 → 3.648 ms`) bo'ldi. Bu kichik launch foydasi, lekin asosiy
one-token gapni yopadigan sakrash emas; dynamic adaptive path uchun defaultga
qo'yilmadi.

`torch.cuda.make_graphed_callables` uchun `router_decisions` va `soft_route`
metadata'lari GPU scalar emas, host metadata sifatida qaytarildi. Bu numerik
model outputini o'zgartirmaydi va router testlari saqlandi.

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
