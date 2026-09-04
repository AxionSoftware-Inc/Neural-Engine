# V0.91 500M Operation-Bank Parent Growth

## Question

V0.90 showed that scratch 500M scaling regresses. This experiment tests a
warm-start path: census the routes of the trained 300M operation-bank model,
copy its shared controller and factor rows into a 500M target, clone the most
used parent factors into the 45 new factor rows, and continue training for
3,000 steps.

The new `grow_dynamic_factorized_capacity.py` utility handles the factorized
operation-bank layout. Because factor_count changes from 154 to 199, this is
an optimization control, not an exact virtual-address-preserving copy.

## Reproduction

```powershell
python -u grow_dynamic_factorized_capacity.py --parent-checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed17_9000.pt --target-config configs/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values.yaml --output results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_growth_init_seed17.pt --device cuda --count-batches 64 --count-batch-size 128 --clone-noise 0.05
python -u train_composition.py --config configs/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values.yaml --steps 3000 --seed 17 --init-checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_growth_init_seed17.pt --run-id ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_growth_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_growth_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u grow_dynamic_factorized_capacity.py --parent-checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values_seed18_9000.pt --target-config configs/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values.yaml --output results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_growth_init_seed18.pt --device cuda --count-batches 64 --count-batch-size 128 --clone-noise 0.05
python -u train_composition.py --config configs/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_broad_values.yaml --steps 3000 --seed 18 --init-checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_growth_init_seed18.pt --run-id ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_growth_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_500m_composition_typed_write_adapter_operation_circuit_banks_rank8_growth_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 78.2715% | 88.1836% | 68.3594% | 9,089,415 | 1,859,488 |
| 18 | 100.0000% | 79.6875% | 91.0156% | 68.3594% | 9,089,415 | 1,859,488 |
| **mean** | **100.0000%** | **78.9795%** | **89.5996%** | **68.3594%** | — | — |

The V0.90 scratch 500M mean is 78.6133%, so parent growth recovers 0.3662
points. However, the V0.79 300M scratch mean is still 79.0771%, so this
simple growth recipe remains 0.0977 points below the smaller model. The
estimated active path stays unchanged at 1,859,488 despite the larger stored
bank.

## Decision

Weak positive for warm-start optimization, not a quality scaling law. Do not
move to 700M or 1B yet. A proper stable growth scheme must preserve factor
semantics/address geometry or train new capacity with an explicit credit
assignment curriculum before larger runs are justified.

The growth utility smoke test used 8 census batches; the measured runs used
64 batches and 128 examples per batch. Checkpoint binaries remain local and
ignored by Git; the commands and results are versioned here.
