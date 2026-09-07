# P-001 — Sparse output-signature follow-up audit

**Sana:** 2026-09-07  
**Branch:** `exp/test-p003-progressive-capacity` (results are local ignored JSONs)  
**Status:** `REJECTED FOR ADOPTION; BANK INITIALIZATION RETAINED AS A HYPOTHESIS`

## Maqsad

Handoff D candidate-only output-signature selectori `E=32, M=8, active=2,
T=3` protokolida sinovdan o‘tdi. Birinchi run’da selector signature’lari
randomdan o‘qitilgan edi. Bu follow-up uch savolni ajratadi:

1. signature rank/dimension oshishi yetarlimi;
2. signature’ni muzlatilgan circuit bank funksiyasidan boshlash foyda beradimi;
3. boshlang‘ich bank signalini keyin moslashtirish kerakmi yoki muzlatish kerakmi.

Retriever, circuit bank, model body va inference active budget o‘zgartirilmadi.
Full-bank circuit inference ishlatilmadi; treatment inference’da faqat tanlangan
ikki real circuit ishladi.

## Natijalar

Har bir variant seed17/18da 5,000 selector step bilan, bir xil held-out
evaluator va oldindan belgilangan gate bilan tekshirildi.

| Variant | Mean Δ hard acc | Mean regret reduction | Mean p95 reduction | Max latency | Selector params | Touched/decision | Qaror |
|---|---:|---:|---:|---:|---:|---:|---|
| Random rank4 / dim16 | `−0.547 pp` | `4.10%` | `8.14%` | `1.042x` | 22,081 | 7,849 | Rejected |
| Random rank8 / dim32 | `−0.026 pp` | `6.03%` | `7.36%` | `1.421x` | 50,273 | 18,761 | Rejected |
| Bank-init rank8 / dim16, trainable | `+0.469 pp` | `6.77%` | `9.43%` | `1.043x` | 40,513 | 12,457 | Rejected for gate |
| Bank-init rank8 / dim16, frozen | `−0.651 pp` | `1.72%` | `3.67%` | `1.190x` | 40,513 | 12,457 | Rejected |
| Bank-init rank8 / dim16, individual-additive target | `−0.234 pp` | `6.51%` | `12.16%` | `1.312x` | 40,513 | 12,457 | Rejected |
| Bank-init rank8 / dim16 + key prior 0.25 | `−0.313 pp` | `3.14%` | `7.77%` | `1.063x` | 40,513 | 13,481 | Rejected |

Bank-init trainable variantning seed-level natijasi:

| Seed | Δ CE | Δ hard accuracy | Mean regret reduction | P95 reduction | Latency |
|---:|---:|---:|---:|---:|---:|
| 17 | `−0.072743` | `+0.469 pp` | `15.71%` | `17.75%` | `0.935x` |
| 18 | `+0.035187` | `+0.469 pp` | `−2.18%` | `1.10%` | `1.043x` |

Qabul gate’i mean hard accuracy `>=+2 pp`, ikkala seedda regressiya yo‘qligi,
mean va p95 regret reduction `>=10%`, recall pasaymasligi va latency `<=1.25x`
edi. Bank-init trainable variant accuracy/latency guardlardan o‘tdi, lekin
accuracy `+0.469 pp` va regret `6.77%/9.43%` bo‘lib gate’dan o‘tmadi.

## Talqin

- Faqat signature capacity’ni oshirish monoton foyda bermadi: rank4/dim16
  yomonroq, rank8/dim32 esa latency va seedlararo barqarorlik muammosiga ega.
- Bankdan tayyor low-rank circuit funksiyasini boshlang‘ich qilish eng kuchli
  signal bo‘ldi. Bu “Qwen neyronlarini ko‘r-ko‘rona ko‘chirish” emas, frozen
  circuit parametrlaridan kichik projected initialization olishdir.
- Signature’ni muzlatish yomon chiqdi; demak bank signalini local pair/GRU
  targetiga moslashtirish kerak. Lekin ikki seeddagi `+0.469 pp` hali kichik
  va regret gate’ini yopmaydi.
- Individual-additive target p95 regretni `12.16%`gacha olib chiqdi, lekin
  final hard accuracy `−0.234 pp` va max latency `1.312x` bo‘ldi. Demak lokal
  pair-selection regretining yaxshilanishi final recurrent cascade sifatiga
  avtomatik ko‘chmaydi.
- Existing key-score’ni `0.25` prior bilan qo‘shish latency guard’dan o‘tdi,
  lekin hard accuracy `−0.313 pp` va regret `3.14%/7.77%` bo‘ldi. Demak
  output-aware signal va retrieval score oddiy additive prior sifatida
  birlashtirilganda ham final sifat muammosi hal bo‘lmaydi.
- Handoff D candidate-only selectori umumiy sifat yechimi sifatida qabul
  qilinmadi. P-001 ochiq qoladi; P-002/P-004 bilan bog‘liq bank
  specialization va cascade target mismatch hali hal qilinmagan.

## Artefaktlar va qayta ishga tushirish

Raw JSONlar Git’dan ignore qilingan `results/runs/` ichida:

- `p001_sparse_output_signature_rank4_dim16_s17_s18.json`
- `p001_sparse_output_signature_rank8_dim32_s17_s18.json`
- `p001_sparse_output_signature_bank_init_rank8_dim16_s17_s18.json`
- `p001_sparse_output_signature_bank_init_freeze_rank8_dim16_s17_s18.json`
- `p001_sparse_output_signature_bank_init_individual_rank8_dim16_s17_s18.json`
- `p001_sparse_output_signature_bank_init_keyprior025_rank8_dim16_s17_s18.json`

Bank-init implementation `exp/p001-bank-initialized-signature` branchida
`3766464` commit sifatida opt-in patch qilib saqlandi; default modelga merge
qilinmadi. Masalan, trainable bank-init run:

```powershell
python -u benchmark_p001_sparse_output_signature.py `
  --checkpoint results/checkpoints/capacity_audit_c32_global_s17.pt `
  --checkpoint results/checkpoints/capacity_audit_c32_global_s18.pt `
  --device cuda --steps 5000 --signature-rank 8 --signature-dim 16 `
  --init-from-bank
```
