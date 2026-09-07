# P-007 — Learned bounded correction gate audit

Sana: 2026-09-07. Maqsad: circuit correction shared reinjection ichida juda
kichik yoki noto‘g‘ri ta’sir qilayotgan bo‘lsa, `step_query + circuit_delta`
asosida o‘rganuvchi bounded gate route’ni sababiyroq va foydaliroq qiladimi?

## Patch

`correction_gate_mode=route_bounded` opt-in varianti qo‘shildi. Gate
`2*sigmoid(linear([step_query, circuit_delta]))` ko‘rinishida va nol weight/bias
bilan boshlanadi, shuning uchun dastlabki correction scale aynan `1.0`. Default
mode `none`; eski model/checkpoint API’si saqlanadi. `expand_checkpoint.py` yangi
gate parametrlarini warm-startda identity initialization bilan qabul qiladi.

## Training protocol

100M seed17: NE-20 parent 5k → 1,408 reachable clamp 5k → full 7,552 bank 5k
→ full continuation 5k. Default 100M staged control bilan bir xil batch,
optimizer va data ishlatildi.

| Variant | Clamp all | Full 5k all | Full 10k all | Held-out active-8 |
|---|---:|---:|---:|---:|
| Default gate=none | 77.92% | 82.99% | **85.57%** | 86.20% |
| Learned route-bounded gate | 77.68% | 83.10% | 85.29% | **86.25%** |

Gate full 10k all-screenda `−0.28 pp`, held-out active-8da esa faqat `+0.05 pp`
berdi. Route replay 100% swap accuracy drop global/within-task `+0.05/+0.10 pp`
bo‘lib, P-007ning `>=+1 pp` sensitivity mezonidan o‘tmadi.

## Qaror

**REJECTED FOR ADOPTION.** Learned gate kichik held-out farq berdi, lekin
all-screen qualityni pasaytirdi va route causalityni sezilarli kuchaytirmadi.
Seed18ni to‘liq trainingga kiritish shart emas: seed17ning o‘zida hard-quality
gate bajarilmadi. Patch va configs opt-in diagnostika sifatida saqlandi,
default gate esa o‘chirildi.

## Reproduction

- Code: `neural_engine/model.py`, `train.py`, `expand_checkpoint.py`.
- Configs: `configs/ne_100_v12_gate_capacity_clamp_b128.yaml`,
  `configs/ne_100_v12_gate_coverage_b128.yaml`.
- Checkpoint: `results/checkpoints/ne100_gate_staged_s17_full_10000.pt`.
- Eval: `results/runs/ne100_gate_active_budget_10k_s17.json` va
  `results/runs/ne100_gate_route_ablation_10k_s17.json`.
