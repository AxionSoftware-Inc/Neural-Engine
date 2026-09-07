# Qwen exact FFN transfer re-check after power recovery

Status: `VALIDATED — NO-TRAINING COMPILATION CONTROL`

## Natija

Lokal cache’dagi Qwen3-0.6B modeli qayta yuklandi. Uning 28 ta MLP qatlami
`gate_proj`, `up_proj`, `down_proj` og‘irliklarini Neural Engine’ning
attention-free `SiLU(gate(x)) * up(x) -> down(x)` circuit formatiga to‘g‘ridan-
to‘g‘ri ko‘chirdim. Hech qanday training yoki calibration correction
ishlatilmadi.

| Metrika | Natija |
|---|---:|
| Model | Qwen3-0.6B |
| Layerlar | 28 |
| Hidden size | 1024 |
| Test input | random, batch 1 × sequence 32 |
| Dtype | float32 |
| Converted parameters | 264,241,152 |
| Max layer MLP error | **0.0** |
| Mean layer MLP error | **0.0** |
| Max full-model logit error | **0.0** |
| Mean full-model logit error | **0.0** |
| Float32 allclose | `true` |

Bu natija oldingi V0.151 exact-transfer controlini tok uzilishidan keyin
mustaqil qayta tasdiqlaydi. Qwen’ning tayyor FFN funksiyasini qayta soatlab
o‘qitish shart emas; vaznlarni mos circuit parametrlariga kompilyatsiya qilish
mumkin.

## Cheklov va keyingi qadam

Bu hali to‘liq Transformer’dan voz kechish emas: Qwen attention bloklari va
qolgan model oqimi o‘z joyida, faqat MLP sublayer almashtirildi. Shuningdek,
to‘liq dense FFN’ni aynan ko‘chirish active compute’ni kamaytirmaydi.

Oldingi held-out text auditida mustaqil neuron guruhlarini 50% yoki 25% aktiv
qilish qualityni buzgan. Shuning uchun keyingi asosiy sinov — Qwen activation
va output contributionlaridan functionally coherent micro-group/adapter
yaratish. Uni avval held-out `data/qwen_eval.txt`da tekshirish kerak; 1Bga
scale qilish bu gate’dan oldin rejalashtirilmaydi.

## Reproduction

```powershell
python benchmark_qwen_transfer.py `
  --model Qwen/Qwen3-0.6B --local-files-only --device cuda `
  --dtype float32 --batch-size 1 --sequence-length 32
```

JSON: `results/runs/qwen_exact_transfer_post_reboot_20260907.json`.
