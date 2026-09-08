# P-004 — State-history skip dataflow audit

Sana: 2026-09-08. Maqsad: single recurrent GRU state oldingi partial resultni
keyingi operation uchun overwrite qilishi mumkin degan gipotezani minimal,
parameter-free history read bilan tekshirish.

## O‘zgarish

`state_history_mode="sum"` opt-in rejimida har bir internal stepdan keyingi
full state historyda saqlanadi. Step-0 va step-1 state’lari allaqachon
`active_state` ichida bo‘lgani uchun keyingi query ularga ko‘paytirilmaydi;
step-2 dan boshlab undan ham oldingi state’lar normalized sum sifatida query’ga
qo‘shiladi. `state_history_scale=0.5` final variantda ishlatildi.

Bu patch router, circuit bank, correction formulasi, active-K va GRU
parametrlarini o‘zgartirmaydi. History tensori parametr emas; serving default
`state_history_mode="none"` bo‘lib qoladi.

## Protocol

20M `coverage_matched_5000` checkpointlarida seed17/18, so‘ng seed19 uchun
5,000 continuation qadamli matching benchmark o‘tkazildi. Har bir run’da
control final loss bilan, treatment esa ayni modelga composition tasklar uchun
`stage_loss_weight=0.1` bilan o‘qitildi. Shu sabab history+stage natijasini
history’siz stage-only bilan alohida solishtirish zarur.

## Natijalar

### 5k history+stage vs final-loss control

| Seed | Δ accuracy | Δ CE | Depth-2+ Δ accuracy | Depth-3 Δ accuracy |
|---:|---:|---:|---:|---:|
| 17 | `+0.260 pp` | `+0.014607` | `+1.172 pp` | `+0.000 pp` |
| 18 | `+1.146 pp` | `−0.015568` | `+2.604 pp` | `+2.865 pp` |
| 19 | `+0.938 pp` | `−0.000646` | `+2.604 pp` | `+3.646 pp` |
| **Mean** | **`+0.781 pp`** | **`−0.000536`** | **`+2.127 pp`** | **`+2.170 pp`** |

### Matching history’siz stage-only vs final-loss control

| Seed | Δ accuracy | Δ CE |
|---:|---:|---:|
| 17 | `+0.573 pp` | `−0.020005` |
| 18 | `−0.260 pp` | `+0.004352` |
| 19 | `+0.104 pp` | `−0.009409` |
| **Mean** | **`+0.139 pp`** | **`−0.008354`** |

History+stage minus stage-only gives an approximate history interaction of
`+0.642 pp` accuracy, but `+0.007819` CE regression. History-only 2k control
also failed (`−0.052/−0.104 pp`, mean `−0.078 pp`), so history is not a
standalone quality fix.

## Task-level signal

At 5k, history+stage improved `reverse_sum` and `chain3` in all three seeds
relative to each run’s control more often than not, and depth-3
`compose_add_mul`/`compose_if` improved especially in seed18/19. `lookup` was
already near solved. `state_machine` remained low and inconsistent, so the
patch did not solve general composition.

## Qaror

**WEAK POSITIVE DIAGNOSTIC, NOT ADOPTED AS DEFAULT.** State history is the
first current-track dataflow patch with a positive three-seed accuracy signal,
but the gain is below the `+2 pp` adoption gate and CE interaction is negative.
The seed-to-seed spread remains large. Keep it opt-in; do not scale to 100M+
or change serving defaults yet. The next experiment should isolate why history
helps some operator chains but not `state_machine`, preferably with
operation-specific state write/read rather than simply adding more history.

## Reproduction

```powershell
python -u benchmark_composition_stage_only.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --steps 5000 --stage-loss-weight 0.1 --state-history-mode sum `
  --state-history-scale 0.5 --eval-batches 4 --eval-examples-per-task 32 `
  --device cuda --output results/runs/state_history_sum05_5k_2x2_ne20_seed17_seed18.json
```

Seed19 and history-only/stage-only matching JSONlar `results/runs/` ichida
saqlangan.
