# Active circuit budget sweep

This sweep keeps the trained NE-V0.12 checkpoint fixed and changes only the
inference-time `active_circuits` top-k budget. It therefore measures the
quality/compute tradeoff of the same learned circuit bank without paying for
three new training runs.

## Reproducibility

```powershell
python analyze_active_budget.py --checkpoint results/checkpoints/ne20_v12_full.pt --device cuda --active-circuits 4 8 16 --examples-per-task 64 --batch-size 128 --iterations 100 --output results/runs/ne20_v12_active_budget_64.json
```

Each quality split contains 64 fresh examples for each of the 15 task IDs
(960 examples). The full split uses all operand combinations; the held-out
split uses the task-aware held-out combination bucket. Inference uses the
learned adaptive halting policy.

## Results

| Active circuits | Full accuracy | Held-out accuracy | Full depth-3 | Held-out depth-3 | Avg active params | Latency | Throughput |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 71.77% | **73.02%** | 32.81% | **35.94%** | 1.957M (9.67%) | **8.281 ms** | **15,457/s** |
| 8 | **71.98%** | 72.71% | 33.33% | 35.42% | 2.039M (10.07%) | 8.441 ms | 15,163/s |
| 16 | 71.88% | 72.92% | **33.33%** | 34.90% | 2.200M (10.87%) | 9.274 ms | 13,802/s |

The unique active parameter estimates are 1.927M (9.52%), 1.978M (9.77%),
and 2.079M (10.27%) for k=4, 8, and 16 respectively. Adaptive execution
remains stable at roughly 1.60 of 3 steps for all budgets.

## Analytical compute trend

| Active circuits | Active MAC estimate | Parameter-read proxy |
|---:|---:|---:|
| 4 | 2.472M/sample | 13.761 MB/sample |
| 8 | 2.551M/sample | 14.085 MB/sample |
| 16 | 2.707M/sample | 14.729 MB/sample |

The 4-circuit setting is not simply “half the compute” of k=8 because the
router, recurrent controller, encoder, and output head are shared. It reduces
the circuit component while leaving the control path in place.

## Decision

The inference-only sweep made k=4 a strong efficiency candidate, but the
follow-up scratch and multi-seed validation is now documented in
`V0_12_MULTI_SEED_BUDGET.md`. It shows that k=4 has competitive held-out
quality but 17.6–21.2% dead circuits, versus 1.85–2.63% for k=8.

**Keep k=8 as the default V0.12 budget** for its more reliable quality and
router coverage. Keep k=4 as an optional low-compute mode until a
coverage-aware low-k training method is tested.
