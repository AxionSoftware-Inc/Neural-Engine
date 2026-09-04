# V0.92 Stable Factor Warm-up for 500M Growth

## Question

V0.91 warm-started a 500M operation-bank model from the 300M parent, but the
larger model still remained below the 300M reference. This experiment tests
whether the regression comes from exposing 45 newly added factor rows too
early. The 500M target starts with only the parent 154 factor rows eligible
for routing, then opens all 199 rows after step 1,000. This is a routing
curriculum, not forced activation: after warm-up, the router remains free to
use any factor it needs.

## Reproduction

```powershell
python -u grow_dynamic_factorized_capacity.py --parent-checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed17_9000.pt --target-config configs/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_broad_values.yaml --output results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_init_seed17.pt --count-batches 64 --count-batch-size 128 --clone-noise 0.05
python -u train_composition.py --config configs/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_broad_values.yaml --steps 3000 --seed 17 --init-checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_init_seed17.pt --run-id ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u grow_dynamic_factorized_capacity.py --parent-checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed18_9000.pt --target-config configs/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_broad_values.yaml --output results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_init_seed18.pt --count-batches 64 --count-batch-size 128 --clone-noise 0.05
python -u train_composition.py --config configs/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_broad_values.yaml --steps 3000 --seed 18 --init-checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_init_seed18.pt --run-id ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_stable_factor_growth_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86. Both runs used
the same 3,000-step, batch-512 benchmark as V0.90 and V0.91.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 80.0293% | 90.1367% | 69.9219% | 9,089,415 | 1,859,488 |
| 18 | 100.0000% | 77.0996% | 86.1328% | 68.0664% | 9,089,415 | 1,859,488 |
| **mean** | **100.0000%** | **78.5645%** | **88.1348%** | **68.9941%** | — | — |

The mean is below V0.90 scratch 500M (78.6133%), V0.91 simple parent growth
(78.9795%), and the V0.80 300M operation-bank reference (79.5898%). The
warm-up did not recover a capacity-quality gain.

## Decision

Reject stable factor warm-up as the default growth method. The result makes
the diagnosis sharper: immediate exposure of new rows is not the only cause
of the 500M regression. More training at the same factorized interface is
unlikely to justify 700M or 1B. The next architecture experiment should
address factor-bank expressivity and ordered factor semantics, then re-run a
small two-seed 300M screen before another scale jump.

Checkpoint binaries and JSON runs remain local and ignored by Git; this report
and its configuration are versioned.
