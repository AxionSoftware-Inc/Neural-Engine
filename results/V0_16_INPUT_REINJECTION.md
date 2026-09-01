# V0.16 recurrent input reinjection ablation

The recurrent update currently receives the encoded input at every internal
step. V0.16 tests whether reducing that reinjection leaves more room for the
persistent state to carry intermediate results. The new
`input_reinjection` parameter scales the original encoded input; the default
remains `1.0`.

## Screening

At 500 steps on the V0.12 NE-20 configuration, `0.5` was slightly better than
the reference while `0.0` was clearly worse:

| Reinjection | Validation | Depth-1 | Depth-3 |
|---:|---:|---:|---:|
| 1.0 reference | 44.11% | 61.59% | 5.47% |
| 0.5 | **45.42%** | **63.89%** | 5.21% |
| 0.0 | 43.15% | 59.68% | 4.17% |

The promising `0.5` setting was then trained for 5000 steps:

```powershell
python train.py --config configs/ne_20_v12_reinject_half.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_reinject_half_checkpoint_5000 --output results/runs --log-every 1000 --checkpoint results/checkpoints/ne20_v12_reinject_half_full.pt
python analyze_active_budget.py --checkpoint results/checkpoints/ne20_v12_reinject_half_full.pt --device cuda --active-circuits 8 --examples-per-task 64 --batch-size 128 --iterations 100 --output results/runs/ne20_v12_reinject_half_active_budget_64.json
```

## Full result

| Reinjection | Validation | Fresh full | Held-out | Full depth-3 | Held-out depth-3 | Dead circuits |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 reference | **71.98%** | **71.98%** | **72.71%** | 33.33% | 35.42% | 1.85% |
| 0.5 | 71.30% | 71.67% | 72.29% | **33.85%** | 35.42% | 3.84% |

The active parameter estimate, adaptive schedule, and inference throughput
remain effectively unchanged. The small depth-3 gain is not enough to offset
the general quality and routing-utilization losses.

## Decision

Keep `input_reinjection: 1.0` as the V0.12 default. Keep the half-reinjection
configs as a reproducible negative ablation; further recurrent-state work
should target an explicit memory/write mechanism rather than simply removing
the input signal.
