# V0.95 Route Utilization Audit

## Question

The 500M screens regress even though the router exposes many more virtual
addresses. This audit checks whether the new factor rows are dead, whether
larger capacity is actually selected, and whether route usage explains the
quality differences. It uses the same held-out composition examples for the
300M parent, 500M scratch, simple parent-growth, and stable-factor-growth
checkpoints.

## Reproduction

```powershell
python -u analyze_composition_routes.py --examples-per-task 1024 --device cuda --output results/route_utilization_audit_20260904.json --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed17_9000.pt --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed18_9000.pt --checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed17_3000.pt --checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed18_3000.pt --checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_growth_seed17_3000.pt --checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_growth_seed18_3000.pt --checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_seed17_3000.pt --checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_seed18_3000.pt
```

## Results

| model screen | held-out mean | virtual circuits used | factor rows used | new rows used (500M) | effective factor rows |
|---|---:|---:|---:|---:|---:|
| 300M parent, seeds 17/18 | 79.59% | 2,271 / 1,967 | 102 / 109 of 154 | — | 60.3 / 59.4 |
| 500M scratch, seeds 17/18 | 78.61% | 1,597 / 1,667 | 147 / 133 of 199 | 35 / 25 | 65.6 / 63.4 |
| 500M simple growth, seeds 17/18 | 78.98% | 2,621 / 2,421 | 111 / 113 of 199 | 39 / 39 | 70.3 / 64.9 |
| 500M stable growth, seeds 17/18 | 78.56% | 2,515 / 2,104 | 124 / 90 of 199 | 44 / 9 | 75.7 / 47.8 |

The new rows are not dead: every 500M run uses between 10 and 40 of the 45
added rows in this held-out audit. However, row usage is not monotonic with
accuracy. The two held-out task route unions also have high factor overlap
(Jaccard about 0.77--0.95), so the router is not cleanly separating the two
operation orders into distinct factor subsets.

## Decision

The evidence rules out “the extra 500M rows are simply never selected” as the
sole explanation. It also rules out route coverage as a sufficient quality
metric. The remaining priority is a capacity-stress task with longer programs
and a better recurrent state/credit-assignment diagnostic before moving to
700M or 1B.
