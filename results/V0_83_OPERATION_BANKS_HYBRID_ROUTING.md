# V0.83 Operation Banks with Hybrid Routing

## Question

V0.80's strongest operation-specific circuit-bank model still has an
asymmetric held-out result, especially for `multiply -> add`. V0.74 showed
that routing from only operation and step metadata discards too much value and
state information. This ablation keeps the full query, but adds a modest
second copy of the current operation and execution-step embedding to the
router query.

The intent was to preserve composition-dependent value information while
making the router's primitive identity more explicit.

## Reproduction

Base source state: `c81df1e` (`reject circuit input normalization`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_hybrid_routing_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_hybrid_routing_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_hybrid_routing_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_hybrid_routing_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_hybrid_routing_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_hybrid_routing_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 99.9721% | 78.2715% | 87.7930% | 68.7500% | 7,361,415 | 1,859,488 |
| 18 | 100.0000% | 76.2207% | 84.1797% | 68.2617% | 7,361,415 | 1,859,488 |
| **mean** | **99.9861%** | **77.2461%** | **85.9863%** | **68.5059%** | — | — |

The V0.79 full-query rank-8 operation-bank baseline reaches 79.0771% mean at
the same 3,000-step budget. Hybrid routing loses 1.8310 points overall and
does not improve the hard `multiply -> add` order. It changes no parameter
count because the extra signal reuses existing embeddings.

## Decision

Rejected as the default. The failure is consistent with V0.74: manipulating
the router query alone is not the missing composition interface. Keep the
full raw query and shared router for now, and test operation-specific router
keys so each primitive can learn its own sparse address geometry.

The full test suite passed after implementation: `86 passed, 2 warnings`.
