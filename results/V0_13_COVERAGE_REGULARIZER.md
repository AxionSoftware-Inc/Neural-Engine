# V0.13 coverage-aware routing regularizer

V0.13 adds an optional training-only auxiliary loss for low active budgets.
The loss builds a differentiable estimate of circuit-bank traffic from the
soft hierarchical tree paths, then minimizes its KL divergence from a uniform
bank distribution. A temperature of `0.25` makes this estimate closer to the
hard argmax route that is used for actual execution.

Normal inference does not enable this calculation. The model still executes
only the selected circuits and keeps the same parameter count and adaptive
halting schedule.

## Reproduction

Both runs used an RTX 3060 12 GB, CUDA, task-balanced batches, and 5000 steps:

```powershell
python train.py --config configs/ne_20_v12_k04_coverage.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_k04_coverage_checkpoint_5000 --output results/runs --log-every 1000 --checkpoint results/checkpoints/ne20_v12_k04_coverage_full.pt
python train.py --config configs/ne_20_v12_k04_coverage_seed18.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_k04_coverage_seed18_checkpoint_5000 --output results/runs --log-every 1000 --checkpoint results/checkpoints/ne20_v12_k04_coverage_seed18_full.pt
```

Fresh full/held-out and latency measurements:

```powershell
python analyze_active_budget.py --checkpoint results/checkpoints/ne20_v12_k04_coverage_full.pt --device cuda --active-circuits 4 --examples-per-task 64 --batch-size 128 --iterations 100 --output results/runs/ne20_v12_k04_coverage_active_budget_64.json
python analyze_active_budget.py --checkpoint results/checkpoints/ne20_v12_k04_coverage_seed18_full.pt --device cuda --active-circuits 4 --examples-per-task 64 --batch-size 128 --iterations 100 --output results/runs/ne20_v12_k04_coverage_seed18_active_budget_64.json
```

## Results

| Budget | Seed | Validation | Full | Held-out | Dead circuits | GPU throughput |
|---:|---:|---:|---:|---:|---:|---:|
| k=4, no coverage | 17 | 71.12% | 70.42% | 72.40% | 21.16% | 15,682/s |
| k=4, coverage | 17 | **72.55%** | **70.83%** | 72.40% | **13.14%** | **15,769/s** |
| k=4, no coverage | 18 | 71.59% | **72.08%** | 72.08% | 17.61% | 14,760/s |
| k=4, coverage | 18 | 71.46% | 71.25% | **72.19%** | **14.84%** | **15,952/s** |

Across the two seeds, coverage-aware training reduces mean dead-circuit rate
from 19.39% to 13.99% and raises mean validation accuracy from 71.36% to
72.01%. Mean held-out accuracy changes from 72.24% to 72.29%; mean fresh full
accuracy changes from 71.25% to 71.04%. Therefore the coverage signal clearly
improves bank utilization, but the current two-seed sample does not establish
a reliable full-benchmark quality gain.

## Decision

Keep `k=8` V0.12 as the default quality/coverage configuration. Keep V0.13
coverage-aware `k=4` as an optional low-compute experiment: it lowers dead
capacity substantially without increasing active parameter estimates, but it
can trade a small amount of full-benchmark accuracy for better router
coverage. The regularizer is disabled by default unless
`routing_coverage_weight` is set in a config.
