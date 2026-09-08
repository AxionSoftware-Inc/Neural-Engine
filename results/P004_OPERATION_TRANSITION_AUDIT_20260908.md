# P-004 — Operation-conditioned low-rank state transition

Sana: 2026-09-08. Maqsad: tanlangan task/operator turiga mos kichik low-rank
state-write transformi Native Engine composition ceilingni pasaytiradimi?

## O‘zgarish

`operation_transition_rank=r` har bir 15 task turi uchun alohida
`state_dim × r × state_dim` emas, faktorlangan `state_dim × r` va
`r × state_dim` adapter saqlaydi. Adapter selected circuit correction,
input reinjection va step embedding yig‘ilgan `update`ga GRU state write’dan
oldin qo‘llanadi:

```text
update -> task-conditioned low-rank transition -> GRUCell -> next state
```

`up` faktor zero-init qilingan, shuning uchun mavjud checkpoint migration’da
birinchi forward control bilan bir xil. Bu output reinjection yoki router
o‘zgarishi emas; state transition matematikasining opt-in o‘zgarishi.

## Protocol

20M `coverage_matched_5000` seed17/18 checkpointlarida control va transition
bir xil task-balanced batchlar, optimizer, 2,000 yoki 5,000 continuation
qadam va `1,920` misollik held-out evaluator bilan solishtirildi. Stage loss
qo‘shilmadi, chunki u alohida auditda final quality gate’dan o‘tmagan.

## Natijalar

| Rank / steps | Seed17 Δ CE | Seed17 Δ acc | Seed18 Δ CE | Seed18 Δ acc | Mean Δ CE | Mean Δ acc |
|---|---:|---:|---:|---:|---:|---:|
| 8 / 2k | `−0.000341` | `+0.417 pp` | `+0.008192` | `−0.469 pp` | `+0.003926` | `−0.026 pp` |
| 32 / 2k | `−0.006168` | `+0.469 pp` | `−0.018924` | `+0.938 pp` | `−0.012546` | `+0.703 pp` |
| 32 / 5k | `−0.026315` | `+0.104 pp` | `−0.021369` | `+0.521 pp` | `−0.023842` | `+0.312 pp` |

Rank-32 CE yaxshilanishi ikki seed va ikki budgetda takrorlandi, ammo hard
accuracy foydasi `+0.312/+0.703 pp` oralig‘ida qolib, `+2 pp` adoption gate’iga
yetmadi. 5k continuationda depth-3 o‘rtacha accuracy controlga nisbatan
`−0.521 pp` bo‘ldi. Shuning uchun CE signalini quality sakrashi deb talqin
qilib bo‘lmaydi; bu P-005 CE/hard-accuracy mismatch bilan mos keladi.

## Qaror

Operation-conditioned low-rank transition `REJECTED FOR ADOPTION`. Rankni
8dan 32ga oshirish CE’ni yaxshilagan, lekin composition hard accuracy’ni
barqaror ko‘tarmagan. Default Native model va active budget o‘zgarmadi; rank64
yoki 700M/1B scaling hozircha asoslanmagan.

P-004 uchun keyingi yo‘l alohida typed transition parameterini ko‘paytirish
emas, value representation va operation semantics’ni bir xil reusable
algebraic registerga bog‘laydigan yangi composition primitive’ni sinashdir.

## Reproduction

```powershell
python -u benchmark_operation_transition.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --rank 8 --steps 2000 --eval-batches 4 --eval-examples-per-task 32 `
  --device cuda --output results/runs/operation_transition_ne20_seed17_seed18.json

python -u benchmark_operation_transition.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --rank 32 --steps 2000 --eval-batches 4 --eval-examples-per-task 32 `
  --device cuda --output results/runs/operation_transition_rank32_ne20_seed17_seed18.json

python -u benchmark_operation_transition.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --rank 32 --steps 5000 --eval-batches 4 --eval-examples-per-task 32 `
  --device cuda --output results/runs/operation_transition_rank32_ne20_seed17_seed18_5000.json
```
