# P-003 native dynamic-width runtime audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Device: CUDA GPU reported by the benchmark (RTX 3060 in the training logs)

## Question

Does the learned K=8/K=16 selector reduce actual latency, or does grouping the
narrow and wide samples into separate dispatches erase the circuit-slot saving?

## Protocol

- The same two 500M stable-prefix checkpoints and their learned width heads
  were loaded.
- A balanced batch of 480 examples was used for every variant.
- Five warm-up calls and 20 synchronized CUDA timing calls were measured with
  diagnostic route statistics disabled.
- Fixed K=8 and fixed K=16 are controls. The learned variant chooses K=8 or
  K=16 per recurrent step and executes only that group.

## Result: mean over three seeds

| Variant | Mean latency | Throughput | Mean active width |
|---|---:|---:|---:|
| Fixed K=8 | 32.39 ms | 14,817 samples/s | 8.00 |
| Fixed K=16 | 55.44 ms | 8,656 samples/s | 16.00 |
| Learned K=8/16 | **39.89 ms** | **12,034 samples/s** | 8.58 |

Relative to fixed K=16, learned width was about **28.1% lower latency** and
about 39.0% higher throughput. Relative to fixed K=8, it was about 23.1%
slower, which is expected because the learned path sends a minority of steps
through the wide route and pays an extra grouped-dispatch launch.

The active-width reduction was about 46.4% of K=16 on this timing batch. The
latency reduction was smaller than the width reduction, so kernel launch and
batch partition overhead are material. This is still a practical speed signal,
not a claim of linear FLOP-to-latency scaling.

## Decision

`POSITIVE RUNTIME SIGNAL — OPT-IN ONLY; KERNEL FUSION OPEN`.

Keep learned dynamic width as the preferred opt-in native candidate. Do not make
it the default until a third seed, longer continuation, and a larger-batch /
single-token profile confirm that the launch overhead remains acceptable. A
future fused grouped-dispatch kernel could close part of the gap between the
46% active-slot saving and the 28% measured latency saving.

## Small-batch guard

With `examples_per_task=1` (batch size 15), the learned checkpoint uses the
configured `dynamic_width_min_batch=32` guard and falls back to one fixed K=16
dispatch. Across three seeds this measured `7.51 ms`, versus `7.33 ms` for
fixed K=16 (`~2.3%` overhead). The earlier unguarded prototype was about `11.8
ms` in this regime, so the guard prevents a meaningful small-batch regression.

## Raw evidence and reproduction

- [Runtime JSON](diagnostic_native_width_runtime_20260910.json)
- [Three-seed runtime JSON](diagnostic_native_width_runtime_all3_20260910.json)
- [Small-batch guarded JSON](diagnostic_native_width_runtime_small_batch_20260910_guarded.json)
- [Runtime benchmark](../benchmark_native_width_runtime.py)
- [Learned-width OOD audit](P003_NATIVE_LEARNED_WIDTH_AUDIT_20260910.md)

```powershell
python benchmark_native_width_runtime.py `
  --checkpoints results/checkpoints/ne500_stable_prefix_active16_s17_3000.pt `
    results/checkpoints/ne500_stable_prefix_active16_s18_3000.pt `
    results/checkpoints/ne500_stable_prefix_active16_s19_3000.pt `
  --warmup 5 --repeats 20 `
  --output results/diagnostic_native_width_runtime_all3_20260910.json
```
