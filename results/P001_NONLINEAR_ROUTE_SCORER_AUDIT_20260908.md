# P-001 — Nonlinear candidate-score residual audit

Sana: 2026-09-08. Maqsad: candidate pool ichidagi oddiy query–key score final
output costni ifodalamasa, query va candidate key’dan kichik nonlinear residual
scorer train qilib selectionni yaxshilash mumkinmi — shuni tekshirish.

## Patch

`HierarchicalRouter.enable_candidate_score_residual(hidden_dim=32)` opt-in
API’si qo‘shildi. Residual MLP `2*state_dim → 32 → 1` bo‘lib, oxirgi qavati
nolga initialize qilindi; boshlang‘ich route va output eski dot-product route
bilan aynan teng. Default router va eski checkpointlar o‘zgarmaydi.

## Protocol

20M `coverage_matched_5000` checkpointlardan seed17/18 uchun control va
treatment bir xil 2,000 continuation qadam o‘qidi:

- batch `128`, task-balanced train, CUDA/RTX 3060;
- training-only soft candidate mixture `temperature=0.5`, ya’ni candidate
  pooldagi 32 circuitning gradienti bor;
- evaluation hard top-8, `adaptive=False`, 1,920 held-out misol;
- control: eski dot-product scorer;
- treatment: eski score + zero-initialized nonlinear residual;
- circuit bank, recurrent body, active inference budget `K=8` bir xil.

Soft training diagnostik signalni barcha candidate circuitlarga yetkazadi,
lekin training xarajati sparse inference’dan kattaroq. Treatment residualida
`24,641` qo‘shimcha parametr bor.

## Natijalar

| Seed | Control CE | Treatment CE | ΔCE | Control acc. | Treatment acc. | Δaccuracy |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.763027 | 0.764179 | +0.001152 | 74.479% | 74.271% | −0.208 pp |
| 18 | 0.734965 | 0.755171 | +0.020205 | 76.250% | 75.677% | −0.573 pp |

Training vaqti seed17/18da `236.1/234.2 s`, peak VRAM `1,434 MB` bo‘ldi.
Inference’da active circuit soni `K=8` bo‘lib qoldi, ammo quality regressiyasi
ikki seedda ham bir xil yo‘nalishda.

## Qaror

**Nonlinear candidate-score residual direct fix sifatida rad qilindi.**
Boshlang‘ich parity to‘g‘ri bo‘lsa ham, soft-routing continuation treatment’i
control’dan yomonlashdi. Bu “nonlinear scorer matematik jihatdan imkonsiz”
degani emas; aynan hozirgi hierarchical candidate pool, soft-training va
final loss bilan u foydali signalga aylanmadi.

Shu bilan birga candidate pool ichida selection headroom, full-bankda esa undan
katta retrieval headroom bor. Keyingi variant yana score boshini kattalashtirish
emas: retrieval va subset selectionni bir vaqtda, final corrected-output
feedback bilan, lekin 32 candidate circuitni trainingda doimiy dense
aktivlashtirmaydigan usulda o‘rgatishi kerak. Hozircha 700M/1Bga o‘tish kerak
emas.

## Artefaktlar

- Router patch: `neural_engine/router.py`.
- Benchmark: `benchmark_nonlinear_route_scorer.py`.
- Unit test: `tests/test_router.py`.
- Raw runs: `results/runs/nonlinear_route_scorer_ne20_s17.json` va
  `results/runs/nonlinear_route_scorer_ne20_s18.json`.
