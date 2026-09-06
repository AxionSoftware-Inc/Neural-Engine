# P-001–P-006 sparse-credit audit — independent recheck

Sana: 2026-09-07. Expert patchi va uning benchmarki Codex tomonidan mustaqil
ravishda qayta ishlatildi. Ekspert hisobotidagi adoption qarori tasdiqlandi:
credit patchi sifat yechimi sifatida qabul qilinmaydi.

## Reproduction

```powershell
python benchmark_sparse_credit_audit.py --device cpu --steps 500 --calibration-steps 200 --train-examples-per-task 4 --eval-batches 16 --capacity --capacity-batches 16 --output results/runs/p001_p006_sparse_credit_audit_recheck_capacity_20260907.json
python -m pytest -q
```

## Paired result: underused-margin vs control

| Seed | Accuracy delta | CE gain | Mean regret reduction | P95 regret reduction | Candidate recall delta |
|---:|---:|---:|---:|---:|---:|
| 17 | +0.104 pp | +0.000438 | +0.859% | +0.668% | +0.0 pp |
| 18 | +0.052 pp | +0.000718 | +1.190% | +0.588% | +0.0 pp |
| **Mean** | **+0.078 pp** | — | — | — | — |

Acceptance talablaridagi +2 pp accuracy va har seed uchun kamida 10% mean/p95
regret yaxshilanishi bajarilmadi. Shuning uchun qaror: **REJECTED**.

## Qo‘shimcha tasdiqlar

- Seed17 eng kam gradient olgan circuitning exposure'i `2 → 144` qadamga oshdi,
  lekin sifatdagi foyda juda kichik qoldi.
- Fresh on-policy calibration static controldan barqaror ustun bo‘lmadi.
- P-003 read-only matched evaluation natijasi: NE20 `73.385%`, NE50 `72.240%`,
  NE100 `73.750%`. Bu checkpointlar matched-training tajribasi emas; 5000 qadam
  yetarliligi va fundamental scaling chegarasi hali aniqlanmagan.
- Full benchmark natijasi `reject_for_adoption`.
- Barcha testlar: `140 passed`, 2 mavjud PyTorch warning.

## Qaror

Patch instrumentation va diagnostic sifatida qoldirilishi mumkin. Router/model
body/defaultlar almashtirilmaydi. P-002 specialization va P-003 matched-capacity
training ochiq qoladi; gradient exposure'ni ko‘paytirishning o‘zi bottleneckni
hal qilmagani tasdiqlandi.
