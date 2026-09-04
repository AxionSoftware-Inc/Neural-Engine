# V0.100 Structured Scalar Value Lane

## Question

The adapter-placement tests did not repair the asymmetric composition
boundary. V0.100 adds a small operation-shared scalar lane whose transition
receives the four features `(old_value, operand, old_value * operand, bias)`.
The four coefficients are learned per primitive, while the scalar state
format is shared across all operations. The lane is injected into the query
and output state without attention or a dense transition table.

Three controls were run:

1. unsupervised scalar lane, injection scale `1.0`;
2. stage-supervised scalar lane, injection scale `1.0`;
3. stage-supervised scalar lane, injection scale `0.1`.

The supervision uses the existing intermediate composition labels only to
test whether the lane can learn a canonical value contract; it is not a
Qwen/Transformer teacher.

## Reproduction

Unsupervised:

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_unsupervised_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_unsupervised_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Supervised scale `1.0` and scale `0.1` used the same commands with their
respective config/run IDs:

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_supervised_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_supervised_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_supervised_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_supervised_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_supervised_low_scale_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_supervised_low_scale_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_supervised_low_scale_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_supervised_low_scale_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_supervised_low_scale_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_structured_scalar_supervised_low_scale_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| variant | seed 17 held-out | seed 18 held-out | two-seed mean | total params | active estimate |
|---|---:|---:|---:|---:|---:|
| unsupervised, scale 1.0 | 79.00% | 71.09% | 75.05% | 7,362,195 | 1,860,268 |
| supervised, scale 1.0 | 77.69% | 73.14% | 75.42% | 7,362,195 | 1,860,268 |
| supervised, scale 0.1 | 74.17% | 78.17% | 76.17% | 7,362,195 | 1,860,268 |
| V0.80 reference | 80.27% | 78.91% | **79.59%** | 7,361,415 | 1,859,488 |

The supervised lane learns nontrivial primitive coefficients, but the
additional signal does not translate into better unseen-order accuracy. Even
at scale `0.1`, the mean remains 3.42 points below V0.80. The lane adds only
780 stored/active estimated parameters, so the failure is architectural
interference rather than a capacity tradeoff.

## Decision

Reject the structured scalar lane as the default. It is not enough to expose
a canonical low-dimensional value path; the surrounding circuit/query/write
system still fails to transfer one operation's representation to an unseen
next operation. Do not scale this lane or add more scalar-loss/scale sweeps
before a new circuit-composition design is specified.

The V0.80 300M operation-bank rank-8 model remains frozen as the learned
reference. The next useful work is a new benchmark or a genuinely different
composition mechanism, not another capacity increase on this state interface.
