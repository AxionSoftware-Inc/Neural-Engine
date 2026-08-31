# V0.12 active-budget multi-seed validation

The first active-budget sweep used one k=8-trained checkpoint and suggested
that k=4 could retain quality at lower active cost. This follow-up trains k=4
and k=8 from scratch with two seeds, then evaluates each checkpoint on the same
fresh full and held-out batches.

## Reproducibility

Training commands:

```powershell
python train.py --config configs/ne_20_v12_k04.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_k04_checkpoint_5000 --checkpoint results/checkpoints/ne20_v12_k04_full.pt --log-every 1000
python train.py --config configs/ne_20_v12_k04_seed18.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_k04_seed18_checkpoint_5000 --checkpoint results/checkpoints/ne20_v12_k04_seed18_full.pt --log-every 1000
python train.py --config configs/ne_20_v12_seed18.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_seed18_checkpoint_5000 --checkpoint results/checkpoints/ne20_v12_seed18_full.pt --log-every 1000
```

The existing `configs/ne_20_v12.yaml` and checkpoint provide the k=8 seed-17
reference. Each checkpoint was then evaluated with:

```powershell
python analyze_active_budget.py --checkpoint <checkpoint.pt> --device cuda --active-circuits <k> --examples-per-task 64 --batch-size 128 --iterations 100
```

## Scratch-training validation

| Budget | Seed | Validation accuracy | Depth-3 accuracy | Dead circuits | Avg active fraction |
|---:|---:|---:|---:|---:|---:|
| k=4 | 17 | 71.12% | 32.94% | 21.16% | 9.67% |
| k=4 | 18 | 71.59% | 34.51% | 17.61% | 9.67% |
| k=8 | 17 | 71.98% | 32.55% | 1.85% | 10.07% |
| k=8 | 18 | **72.19%** | 34.11% | 2.63% | 10.07% |

Across the two seeds, k=8 is 0.73 percentage points higher on mean validation
accuracy and has a much healthier circuit-bank utilization profile. Both
budgets learn the adaptive schedule of approximately 1.60/3 executed steps.

## Fresh full and held-out evaluation

These values come from the same 64-examples-per-task evaluation procedure used
by `analyze_active_budget.py`, rather than the training script's separate
validation stream.

| Budget | Seed | Full accuracy | Held-out accuracy | Full depth-3 | Held-out depth-3 | GPU throughput |
|---:|---:|---:|---:|---:|---:|---:|
| k=4 | 17 | 70.42% | 72.40% | 31.25% | 34.90% | 15,682/s |
| k=4 | 18 | 72.08% | 72.08% | 34.90% | 32.29% | 14,760/s |
| k=8 | 17 | 71.98% | **72.71%** | 33.33% | **35.42%** | 15,163/s |
| k=8 | 18 | **72.40%** | 71.77% | 33.33% | 28.65% | 14,088/s |

Mean held-out accuracy is 72.24% for both budgets on these two seeds. The
sample is useful but not large enough to claim a definitive generalization
winner; the training-validation and circuit-utilization results are more
consistent in favor of k=8.

## Decision

**Keep k=8 as the default V0.12 budget.** It costs only about 0.40 percentage
points more average active parameter fraction than k=4, but its scratch-trained
routers use far more of the bank and deliver a more reliable full-benchmark
quality signal across seeds.

Keep k=4 as an optional low-compute mode. Its held-out quality is competitive,
and its analytical path cost is slightly lower, but the 17.6–21.2% dead-circuit
rate shows that the smaller budget currently underuses the learned capacity.
Before making k=4 the default, improve low-k router coverage or train it with a
coverage-aware regularizer.

