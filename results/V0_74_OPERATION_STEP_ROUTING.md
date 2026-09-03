# V0.74 Operation-Step Routing Ablation

## Question

The default `full` route query depends on the current accumulator and operand
state. That can make a held-out operation order select a route that was never
trained for the new composition. This ablation forces routing to depend only
on the primitive operation and execution step, so the same circuit route is
reused across different value states and composition orders.

The typed-write adapter, learned value encoder, factorized 23,600-circuit
bank, active `k=8` path, and broad non-modular benchmark are unchanged.

## Reproduction

Base commit: `9e954cf` (`reject scalar gaussian output ablation`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_step_routing_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_step_routing_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_step_routing_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_step_routing_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_step_routing_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_step_routing_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 68.2617% | 71.0938% | 65.4297% | 3,495,299 | 1,693,584 |
| 18 | 100.0000% | 66.6016% | 69.5313% | 63.6719% | 3,495,299 | 1,693,584 |
| **mean** | **100.0000%** | **67.4316%** | 70.3125% | 64.5508% | — | — |

The V0.69 full-route typed-write baseline is 69.1650% on the same 3,000-step
screen. Operation-step routing therefore loses 1.7334 points. It does not
show a routing-collapse fix; removing value/state context from the router
also removes useful information needed by the circuit bank.

## Decision

Rejected as the default routing policy. Keep `route_context_mode: full` for
the current learned reference. The result does not justify growing the model,
adding more route levels, or hard-coding operation-only routes: the broad
quality gap remains an internal numeric composition problem.

The full test suite passed before this config-only experiment: `83 passed, 2
warnings` (the existing Transformer nested-tensor warnings).
