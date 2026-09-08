# P-004 — State-only intermediate supervision audit

Sana: 2026-09-08. Maqsad: oldingi stage-loss tajribasida partial targetlar
final `output` head’i bilan aralashib ketgan bo‘lishi mumkin degan gipotezani
tekshirish. Yangi treatment intermediate targetni recurrent state’dan o‘qiydigan
alohida auxiliary head bilan o‘rgatadi; asosiy final output head’i bu lossni
olmaydi.

## O‘zgarish

Opt-in `state_stage_head=True` `state_dim → num_classes` yordamchi head’ini
qo‘shadi. U faqat depth-2/3 tasklarning deterministic stage targetlari uchun
training paytida ishlatiladi. `collect_stats=False` serving yo‘lida head
hisoblanmaydi va final logitsga to‘g‘ridan-to‘g‘ri qo‘shilmaydi. Treatment armga
`25,408` total/active-estimate parametr qo‘shildi.

## Protocol

20M `coverage_matched_5000` seed17/18 checkpointlarida control faqat final CE
loss oldi; treatment esa final CE + state-only stage loss (`weight=0.1`) oldi.
Har ikkisi bir xil task-balanced batch stream, 2,000 continuation qadam va
1,920 misollik held-out evaluator bilan tekshirildi.

## Natijalar

| Seed | Δ accuracy | Δ CE | Δ stage-0 | Δ stage-1 | Δ stage-2 |
|---:|---:|---:|---:|---:|---:|
| 17 | `−0.052 pp` | `+0.011910` | `−0.677 pp` | `−0.651 pp` | `−0.260 pp` |
| 18 | `−0.313 pp` | `+0.004274` | `+0.104 pp` | `−1.042 pp` | `+1.562 pp` |
| **Mean** | **`−0.182 pp`** | **`+0.008092`** | **`−0.286 pp`** | **`−0.846 pp`** | **`+0.651 pp`** |

Depth-3 tasklarda `compose_add_mul`, `compose_if` va `state_machine` natijalari
seedlar bo‘yicha qarama-qarshi qoldi. State head lossining o‘zi pasaygani
intermediate state foydali bo‘ldi degani emas; final quality va stage-0/1
signal buni tasdiqlamadi.

## Qaror

State-only stage head **REJECTED FOR ADOPTION**. Stage supervisionni output’dan
state’ga ko‘chirish ham composition ceilingni ochmadi. Default model va serving
path o‘zgarmadi; auxiliary head opt-in diagnostik kod sifatida qoldi. P-004
uchun keyingi ish yana loss/head varianti emas, reusable algebraic
value/state primitive yoki circuit specializationning forward contractini
tekshirish bo‘ladi. 700M/1B ga scale qilinmaydi.

## Reproduction

```powershell
python -u benchmark_state_stage_head.py `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_5000.pt `
  --checkpoint results/checkpoints/ne20_v12_coverage_matched_seed18_5000.pt `
  --steps 2000 --state-stage-loss-weight 0.1 `
  --eval-batches 4 --eval-examples-per-task 32 --device cuda `
  --output results/runs/state_stage_head_2x2_ne20_seed17_seed18.json
```
