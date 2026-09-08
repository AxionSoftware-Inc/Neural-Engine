# P-004 — Task-scaled state-history audit

Sana: 2026-09-08. Maqsad: oldingi `state_history_mode="sum"` sinovida
history barcha tasklarga bir xil koeffitsient bilan qo‘shilgani sababli ayrim
operatorlar foyda ko‘rmayotgan bo‘lishi mumkin degan gipotezani tekshirish.

## Gipoteza va patch

`state_history_mode="task_scaled"` rejimida 15 ta taskning har biri uchun
bitta o‘rganiladigan history read-scale qo‘shildi. Ular `state_history_scale`
qiymatiga teng boshlanadi; shu sababli boshlang‘ich forward fixed
`state_history_mode="sum"` bilan bir xil. O‘qitish davomida har task historyni
qanchalik o‘qishini mustaqil moslashtirishi mumkin.

Patch router, circuit bank, correction formulasi, active-K, GRU yoki serving
defaultni o‘zgartirmaydi. Qo‘shimcha sig‘im atigi 15 ta scalar parametr.
Checkpoint loader eski checkpointda kutiladigan yagona missing key —
`state_history_task_scales` — ni opt-in rejimda qabul qiladi; boshqa mismatch
rad etiladi.

## Protocol

20M `coverage_matched_5000` checkpointlari, seed17/18, 2,000 continuation
step, `stage_loss_weight=0.1`, bir xil held-out evaluation va fixed
`state_history_scale=0.5` baseline bilan matching comparison ishlatildi.
Treatment `task_scaled`, control esa history-free final-loss control bo‘ldi.
Bu kichik screening tajribasi; adoptiondan oldin kamida 3 seed va 5k
continuation talab qilinadi.

## Natijalar

| Seed | Control accuracy | Treatment accuracy | Δ accuracy | Δ CE |
|---:|---:|---:|---:|---:|
| 17 | 74.3229% | 75.2083% | `+0.885 pp` | `−0.00000399` |
| 18 | 76.1979% | 75.4688% | `−0.729 pp` | `+0.00005757` |
| **Mean** | — | — | **`+0.078 pp`** | **`+0.00002679`** |

Task-stage deltalari ham seedlar orasida qarama-qarshi bo‘ldi:

| Seed | Stage-0 | Stage-1 | Stage-2 |
|---:|---:|---:|---:|
| 17 | `+7.188 pp` | `+1.823 pp` | `+2.344 pp` |
| 18 | `+6.875 pp` | `−0.260 pp` | `−1.563 pp` |

## Qaror

**REJECTED FOR ADOPTION.** Task-specific read-scale stage-0 signalini
oshirgan bo‘lsa ham, final quality bo‘yicha o‘rtacha foyda `+2 pp` adoption
gate’dan juda uzoq, seed sign-flip qildi va mean CE yaxshilanmadi. Demak
muammo historyni tasklar bo‘yicha bitta scalar bilan bo‘lishda emas yoki bu
patch kerakli operator/state dataflowni ifodalay olmaydi.

Bu natija task-scaled variantni defaultga kiritmaslik va uni 100M/300M/500M
ga ko‘chirmaslik uchun yetarli. Opt-in API va test saqlanadi. P-004 `ACTIVE`:
keyingi ish history koeffitsientlarini ko‘paytirish emas, operation-specific
state write/read yoki circuitning intermediate value’ni keyingi operationga
aniq uzatish mexanizmini tekshirishi kerak.

## Reproduction

```powershell
python -u benchmark_composition_stage_only.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --steps 2000 --stage-loss-weight 0.1 `
  --state-history-mode task_scaled --state-history-scale 0.5 `
  --eval-batches 4 --eval-examples-per-task 32 `
  --device cuda `
  --output results/runs/state_history_task_scaled05_2k_2x2_ne20_seed17_seed18.json
```

Raw result: `results/runs/state_history_task_scaled05_2k_2x2_ne20_seed17_seed18.json`.
