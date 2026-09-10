# P-001 — On-policy target retrieval aggregation audit

Sana: 2026-09-10. Bu offline final-CE target distillation regressiyasidan
keyingi follow-up. Maqsad — router o‘zgargandan keyin query/state distribution
siljishi sabab bo‘lsa, har roundda yangi natural route’lardan target yig‘ish
muammoni tuzatadimi, tekshirish.

## Protokol

- Frozen 20M `v12_coverage_matched` seed17/18 checkpointlar;
- `candidate_pool=32`, `active_circuits=8`, `internal_steps=3`;
- har round: 2 train batch × 16 example/task bilan one-swap final-CE target
  yig‘ish;
- 3 round, har roundda 500 AdamW qadam, batch size 2,048;
- yangi target rows eski rows bilan aggregate qilindi;
- faqat hierarchical `level_projections`, `level_bias` va `keys` train qilindi;
- circuit banki, recurrent body, correction va hard inference budget muzlatildi;
- held-out: 1 batch × 16 example/task.

## Natijalar

| Seed | Baseline CE / acc | On-policy treatment CE / acc | ΔCE | Δaccuracy | Treatment unique circuits |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.809630 / 75.83% | 0.828646 / 75.42% | +0.019016 | −0.42 pp | 1105 |
| 18 | 0.805155 / 75.83% | 0.816745 / 73.33% | +0.011590 | −2.50 pp | 1155 |
| **Mean** | — | — | **+0.015303** | **−1.46 pp** | — |

Target signal roundlar davomida saqlanib qoldi: seed17/18da targetlarning
taxminan `68–76%`i current candidate pooldan tashqarida bo‘ldi va round-2/3
one-swap headroomi oshdi. Shunga qaramay held-out hard route sifati tiklanmadi;
router selected bankni toraytirdi va final CE yomonlashdi.

## Qaror

**REJECTED FOR ADOPTION.** On-policy data aggregation frozen circuit body bilan
retrieval targetlarini end-to-end foydaga aylantirmadi. Bu P-001 uchun yana bir
muhim ajratish beradi: muammo faqat stale calibration emas. Router target group
ga o‘rgatilganda natural query, route weight va keyingi recurrent state birga
o‘zgaradi; faqat tree/key parametrlarini qayta o‘qitish bu yopiq siklni
barqarorlashtirmadi.

Shu oiladagi quyidagi variantlar yopildi:

- query/key + output-logit post-hoc surrogate;
- frozen one-swap final-CE target distillation;
- on-policy target aggregation bilan frozen-body retrieval training.

P-001 ochiq qoladi, lekin keyingi ishni yana router feature’lariga sarflash
to‘g‘ri emas. Navbatdagi Native Engine yo‘li circuit output composition va
recurrent state interface’ini body+router bilan birga, kichik 20M nazoratda
tekshirishdir. U ham defaultni o‘zgartirmaydigan opt-in tajriba bo‘ladi.

## Reproduksiya

```powershell
python benchmark_target_retrieval_distill.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --train-batches 2 --eval-batches 1 --examples-per-task 16 `
  --global-topk 8 --rounds 3 --steps 500 --batch-size 2048 `
  --learning-rate 3e-4 --device cuda `
  --output results/runs/target_retrieval_onpolicy_ne20_s17.json

python benchmark_target_retrieval_distill.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --train-batches 2 --eval-batches 1 --examples-per-task 16 `
  --global-topk 8 --rounds 3 --steps 500 --batch-size 2048 `
  --learning-rate 3e-4 --device cuda `
  --output results/runs/target_retrieval_onpolicy_ne20_s18.json
```

Artefakt: `benchmark_target_retrieval_distill.py` (`--rounds 3`).
