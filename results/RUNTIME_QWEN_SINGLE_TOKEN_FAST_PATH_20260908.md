# Runtime — Qwen one-token selected-dispatch fast path

**Sana:** 2026-09-08  
**Branch:** `exp/track-runtime`  
**Qaror:** `PROMISING OPT-IN`; full serving defaulti uchun yana quality/runtime
audit kerak.

## Gipoteza

Qwen grouped dispatch ko'p tokenli batch uchun `sort → bincount → pack → bmm →
scatter` yo'lidan foydalanadi. One-token decode'da bu packing xarajati
selected projectionning o'zidan katta bo'lishi mumkin. `single_token_fast_path`
opt-in flag'i `tokens == 1` bo'lganda faqat tanlangan K group weightlarini
gather qilib, uchta batched contraction bilan outputni hisoblaydi.

Default grouped path o'zgartirilmadi: flag `False`. Fast path circuit tanlovi,
K=6 active budget yoki hard scale'ni o'zgartirmaydi. Reduction ordering
`F.linear`/grouped bmm'dan farq qilishi mumkinligi sababli full-model logit
farqi alohida qayd qilindi.

## Isolated real Qwen layer

Qwen3-0.6B, layer 26, hidden shape `[1, 1, 1024]`, float32, RTX 3060:

| Dispatch | Latency | Fast pathga nisbatan |
|---|---:|---:|
| Existing token-loop | 2.230 ms | 6.11x slower |
| Opt-in grouped single-token | 0.365 ms | 1.00x |

The selected-output max absolute difference versus token-loop was
`1.678e-4`. This is a dispatch-level reduction-order difference; it is not a
quality result.

## Full Qwen forward smoke

One-token full-model forward, `use_cache=False`, same local Qwen checkpoint,
with eight replaced layers `[0,4,8,12,16,20,24,26]`:

| Model path | Latency |
|---|---:|
| Dense parent | 29.725 ms |
| Sparse grouped, old path | 36.294 ms |
| Sparse grouped, fast path | 32.560 ms |

Fast/grouped ratio was `0.897x`, a `10.3%` dispatch improvement. Sparse
fast/parent ratio remained `1.095x`, so attention and routing/controller
overhead still dominate the end-to-end one-token result. Full-model logits
fast versus old grouped had max error `7.629e-6`, mean error `1.247e-6`.

The child in this smoke used an untrained zero-initialized router and was
created for runtime measurement only. Therefore these numbers do not make a
new Qwen quality claim and cannot replace the trained K=5/K=6 quality reports.

## Qaror

- Keep `single_token_fast_path` as an opt-in runtime feature.
- Do not call it a quality-equivalent replacement until a trained multilayer
  checkpoint is evaluated on the existing CE/accuracy gate.
- Do not claim one-token decode is solved: end-to-end sparse path is still
  slower than dense in this smoke.
- Next runtime work is to reduce router/controller launches or provide a
  fused kernel; a simple static CUDA Graph is already only a small gain.

## Reproduksiya

```powershell
python benchmark_qwen_single_token_dispatch.py --layer 26 --warmup 20 --iterations 100
python benchmark_qwen_single_token_model.py --layers 0,4,8,12,16,20,24,26 --warmup 20 --iterations 30
```

Implementation: `benchmark_qwen_multi_layer_transplant.py`; fast path is
enabled explicitly by `benchmark_qwen_single_token_dispatch.py` and
`benchmark_qwen_single_token_model.py`.
