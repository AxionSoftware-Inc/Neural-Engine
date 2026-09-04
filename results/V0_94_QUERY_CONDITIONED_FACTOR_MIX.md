# V0.94 Query-Conditioned Factor Mix

## Question

The factorized circuit bank currently combines two selected factor rows with
the same shared coefficients for every query. V0.94 adds a small learned gate
key per factor row, so the current state can modulate each selected factor's
contribution. It keeps the route sparse: only the factor rows attached to the
selected virtual circuits are read.

## Reproduction

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_query_factor_mix_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_query_factor_mix_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_query_factor_mix_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_query_factor_mix_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_query_factor_mix_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_query_factor_mix_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 77.7832% | 87.7930% | 67.7734% | 7,538,823 | 1,871,776 |
| 18 | 100.0000% | 79.1504% | 89.4531% | 68.8477% | 7,538,823 | 1,871,776 |
| **mean** | **100.0000%** | **78.4668%** | **88.6230%** | **68.3105%** | — | — |

The shared-mix V0.79 reference is 79.0771%. The query-conditioned gate loses
0.6104 points while adding 177,408 stored parameters and 12,288 to the active
estimate. The `multiply -> add` order remains the bottleneck.

## Decision

Reject as the default. A query-dependent factor coefficient is not enough to
turn virtual pair capacity into better held-out composition. The code remains
available as an optional ablation, but the next work should focus on the
recurrent state/task interface and on a harder capacity-stress benchmark.
