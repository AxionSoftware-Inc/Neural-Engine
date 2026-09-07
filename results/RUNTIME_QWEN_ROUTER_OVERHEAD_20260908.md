# Runtime — Qwen one-token router overhead ablation

**Sana:** 2026-09-08  
**Branch:** `exp/track-runtime`  
**Qaror:** `ROUTER IS NOT THE ONLY BOTTLENECK`

## Maqsad

Oldingi fast-dispatch smoke’da Qwen one-token sparse yo‘li dense parentdan
sekinroq qoldi. Bu sinov router MLP xarajatini alohida ajratadi: odatiy
routerning yakuniy logitlari yangi child’da nol bo‘lgani uchun, uning o‘rniga
har bir qatlamda bir xil nol-logit qaytaruvchi statik controller qo‘yildi.
Shu sabab top-k subset, softmax weight va selected groups aynan bir xil qoladi;
faqat router MLP kernel launchlari olib tashlanadi.

Bu runtime-only ablation. Router o‘qitilmagan va quality xulosasi bermaydi.

## Natija

Qwen3-0.6B, float32, RTX 3060, one-token `use_cache=False`, sakkizta
almashtirilgan layer `[0,4,8,12,16,20,24,26]`, grouped single-token fast path,
30 timing iteration:

| Yo‘l | Latency |
|---|---:|
| Dense parent | `29.019 ms` |
| Sparse + odatiy router | `37.245 ms` |
| Sparse + statik nol-logit controller | `33.822 ms` |

Router MLPning taxminiy ulushi `3.423 ms` (`9.2%` of regular sparse path).
Ammo controller olib tashlangandan keyin ham sparse yo‘l dense parentdan
`1.166x` sekin. Static va regular yo‘llarning logitlari aynan teng chiqdi
(`max error = 0.0`, `mean error = 0.0`).

## Xulosa

Router muhim overhead, lekin asosiy muammo faqat router emas. Qolgan taxminan
`4.8 ms` farq selected FFN dispatch, residual/controller launchlari va full
Transformer yo‘lidagi boshqa kernel xarajatlaridan keladi. Oldingi full-model
profilerda scaled-dot-product attentionning o‘zi taxminan `25.5 ms` CUDA self
time olgani ham ko‘ringan. Shu sabab faqat router MLPni ixchamlashtirish bilan
Qwen one-token sparse serving dense’dan tez bo‘lib qolmaydi.

Bu natija attention-free Native Engine yo‘li noto‘g‘ri ekanini ko‘rsatmaydi;
aksincha, Qwen lane’da attention doimiy xarajat ekanini ko‘rsatadi.

## Qaror

- Routerni yengillashtirish foydali mikro-optimallashtirish sifatida qayd
  qilindi, lekin alohida sifat yoki serving yechimi sifatida qabul qilinmadi.
- `single_token_fast_path` grouped varianti opt-in qoladi.
- Keyingi Qwen runtime ishi faqat fused router+top-k+selected-dispatch
  kerneliga o‘tilganda ma’noli bo‘ladi; oddiy router almashtirish yetarli emas.
- Native Engine quality/scaling tajribalari bu Qwen runtime natijalaridan
  alohida baholanadi.

## Reproduksiya

```powershell
python benchmark_qwen_router_overhead.py --layers 0,4,8,12,16,20,24,26 --warmup 20 --iterations 30
```

