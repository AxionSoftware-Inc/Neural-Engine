# V0.14 stage supervision ablation

V0.14 re-tests the existing intermediate-result supervision with the current
V0.12 numeric encoder and adaptive halting. During training, recurrent outputs
are also matched to deterministic partial results supplied by the synthetic
task generator. Inference remains unchanged and still uses only the final
output.

## Reproduction

Both runs used an RTX 3060 12 GB, CUDA, task-balanced batches, and 5000 steps:

```powershell
python train.py --config configs/ne_20_v12_stage.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_stage_checkpoint_5000 --output results/runs --log-every 1000 --checkpoint results/checkpoints/ne20_v12_stage_full.pt
python train.py --config configs/ne_20_v12_stage_seed18.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_stage_seed18_checkpoint_5000 --output results/runs --log-every 1000 --checkpoint results/checkpoints/ne20_v12_stage_seed18_full.pt
```

Fresh benchmark measurements:

```powershell
python analyze_active_budget.py --checkpoint results/checkpoints/ne20_v12_stage_full.pt --device cuda --active-circuits 8 --examples-per-task 64 --batch-size 128 --iterations 100 --output results/runs/ne20_v12_stage_active_budget_64.json
python analyze_active_budget.py --checkpoint results/checkpoints/ne20_v12_stage_seed18_full.pt --device cuda --active-circuits 8 --examples-per-task 64 --batch-size 128 --iterations 100 --output results/runs/ne20_v12_stage_seed18_active_budget_64.json
```

## Results

| Training | Seed | Validation | Full | Held-out | Full depth-3 | Held-out depth-3 |
|---|---:|---:|---:|---:|---:|---:|
| V0.12, no stages | 17 | **71.98%** | **71.98%** | **72.71%** | 33.33% | 35.42% |
| V0.14, stage loss 0.1 | 17 | 71.28% | 71.88% | 71.77% | 35.94% | **35.42%** |
| V0.12, no stages | 18 | **72.19%** | **72.40%** | **71.77%** | 33.33% | 28.65% |
| V0.14, stage loss 0.1 | 18 | 71.67% | 70.83% | 71.15% | **36.98%** | 31.77% |

Across two seeds, stage supervision raises mean full depth-3 accuracy from
33.33% to 36.46% and mean held-out depth-3 accuracy from 32.03% to 33.59%.
However, mean full accuracy falls from 72.19% to 71.35%, and mean held-out
accuracy falls from 72.24% to 71.46%. The auxiliary targets also add a small
training-time cost and do not improve the learned halting schedule.

## Decision

Do not enable stage supervision in the default V0.12 configuration. Keep
`stage_loss_weight: 0.1` as an optional composition-focused recipe when
depth-3 performance matters more than overall or held-out accuracy. This is a
useful specialization signal, not a general quality improvement.
