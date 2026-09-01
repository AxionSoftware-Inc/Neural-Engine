# V0.17 gated memory/write ablation

V0.17 adds an optional learned write gate around the GRU proposal. The gate
sees the current state and the new update, then chooses how much of the
proposal to write back into recurrent memory. It is initialized close to an
identity write so the reference path remains the starting point.

## Reproduction

```powershell
python train.py --config configs/ne_20_v12_memory_write.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_memory_write_checkpoint_5000 --output results/runs --log-every 1000 --checkpoint results/checkpoints/ne20_v12_memory_write_full.pt
python analyze_active_budget.py --checkpoint results/checkpoints/ne20_v12_memory_write_full.pt --device cuda --active-circuits 8 --examples-per-task 64 --batch-size 128 --iterations 100 --output results/runs/ne20_v12_memory_write_active_budget_64.json
```

The run used an RTX 3060 12 GB, CUDA, seed 17, task-balanced batches, and
5000 steps. The baseline is the existing V0.12 checkpoint with no extra write
gate.

## Results

| Variant | Total params | Avg active params | Active fraction | Validation | Fresh full | Held-out | Full depth-3 | Held-out depth-3 | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0.12 reference | 20.25M | 2.039M | 10.07% | **71.98%** | **71.98%** | 72.71% | **33.33%** | 35.42% | **15,163/s** |
| V0.17 gated write | 20.54M | 2.334M | 11.36% | 71.61% | 70.73% | **73.23%** | 29.69% | **39.06%** | 12,590/s |

The gate gives a positive held-out signal, but it lowers in-distribution and
depth-3 full accuracy while increasing active parameter traffic by about 13%
and reducing measured throughput by about 17%. A 500-step screening run also
started below the reference (`42.89%` versus `44.11%` validation accuracy).

## Decision

Do not enable gated memory/write in the default V0.12 configuration. The
result suggests that preserving state can help a particular distribution shift,
but a generic write gate is too expensive and does not improve the main task.
Future memory work should use a cheaper structured or slot-level write
mechanism and should be evaluated on the composition benchmark first.
