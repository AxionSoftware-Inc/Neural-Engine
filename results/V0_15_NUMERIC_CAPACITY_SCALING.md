# V0.15 numeric capacity scaling with coverage-aware routing

This experiment tests the current V0.12 numeric/adaptive model while growing
the dormant circuit bank from NE-20 to NE-100. The active budget stays at
eight circuits per recurrent step, with three maximum steps and adaptive
halting. NE-50 and NE-100 use the optional low-temperature coverage-aware
training loss from V0.13; inference does not evaluate that loss.

## Reproduction

All runs used an RTX 3060 12 GB, CUDA, seed 17, task-balanced batches, batch
size 128, and 5000 steps:

```powershell
python train.py --config configs/ne_50_v12_coverage.yaml --steps 5000 --device cuda --balanced-train --run-id ne50_v12_coverage_checkpoint_5000 --output results/runs --log-every 1000 --checkpoint results/checkpoints/ne50_v12_coverage_full.pt
python train.py --config configs/ne_100_v12_coverage.yaml --steps 5000 --device cuda --balanced-train --run-id ne100_v12_coverage_checkpoint_5000 --output results/runs --log-every 1000 --checkpoint results/checkpoints/ne100_v12_coverage_full.pt
```

Fresh full/held-out and latency measurements used 64 examples per task and
100 latency iterations:

```powershell
python analyze_active_budget.py --checkpoint results/checkpoints/ne50_v12_coverage_full.pt --device cuda --active-circuits 8 --examples-per-task 64 --batch-size 128 --iterations 100 --output results/runs/ne50_v12_coverage_active_budget_64.json
python analyze_active_budget.py --checkpoint results/checkpoints/ne100_v12_coverage_full.pt --device cuda --active-circuits 8 --examples-per-task 64 --batch-size 128 --iterations 100 --output results/runs/ne100_v12_coverage_active_budget_64.json
```

As a follow-up seed check, NE-100 was also trained with:

```powershell
python train.py --config configs/ne_100_v12_coverage_seed18.yaml --steps 5000 --device cuda --balanced-train --run-id ne100_v12_coverage_seed18_checkpoint_5000 --output results/runs --log-every 1000 --checkpoint results/checkpoints/ne100_v12_coverage_seed18_full.pt
python analyze_active_budget.py --checkpoint results/checkpoints/ne100_v12_coverage_seed18_full.pt --device cuda --active-circuits 8 --examples-per-task 64 --batch-size 128 --iterations 100 --output results/runs/ne100_v12_coverage_seed18_active_budget_64.json
```

## Results

| Model | Total params | Avg active params | Active fraction | Validation | Fresh full | Held-out | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| NE-20 V0.12, no coverage | 20.25M | 2.039M | 10.07% | 71.98% | **71.98%** | 72.71% | **15,163/s** |
| NE-50 V0.12 + coverage | 50.33M | 2.038M | 4.05% | **72.03%** | 70.63% | 71.35% | 15,143/s |
| NE-100 V0.12 + coverage | 100.47M | 2.041M | **2.03%** | **72.14%** | 71.35% | **72.92%** | 14,568/s |

### Follow-up seed check

| Model | Seed | Validation | Fresh full | Held-out | Avg active params | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| NE-20 V0.12 reference | 18 | 72.19% | 72.40% | 71.77% | 2.039M | 14,088/s |
| NE-100 V0.12 + coverage | 18 | 72.50% | **73.54%** | **72.60%** | 2.042M | 14,182/s |

The active estimate stays within 0.15% of 2.04M while stored capacity grows
4.96x from NE-20 to NE-100. Adaptive execution remains aligned at about 1.60
of 3 steps. Inference throughput declines gradually as the larger model causes
more memory pressure, but it remains close to the NE-20 throughput because the
selected circuit path is unchanged.

## Routing coverage

The training reports saw 94.2% of NE-50 circuits and 93.8% of NE-100 circuits
on their validation streams. A separate fresh route probe with 64 examples per
task saw 75.4% of NE-50 and 66.3% of NE-100 circuits. This gap shows that the
router is input-sensitive and that a small route sample should not be treated
as a complete bank-utilization estimate.

## Decision

This is a positive fixed-active scaling signal: capacity can grow from 20M to
100M parameters while the active path remains about 2M parameters and held-out
quality does not collapse. Across the two available NE-20/NE-100 seeds, mean
full accuracy rises from 72.19% to 72.45% and mean held-out accuracy rises from
72.24% to 72.76%. NE-50 still has only one seed, so the result is promising
rather than definitive evidence that dormant capacity alone improves quality.
The next priority is a larger, composition-focused benchmark and more seeds
before any custom sparse CUDA kernel work.
