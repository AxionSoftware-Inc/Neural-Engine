# V0.96 Dynamic Depth Capacity Screen

## Question

The two-operation held-out benchmark may not stress recurrent capacity enough.
V0.96 trains on variable-length programs of depths 1--4 and evaluates unseen
depths 5--8. The same operation-specific factorized bank is tested at 300M
and 500M virtual capacity, with 8 active circuits per executed step.

## Reproduction

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_depth8_operation_circuit_banks_rank8.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_depth8_operation_circuit_banks_rank8_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_depth8_operation_circuit_banks_rank8_seed17_3000.pt --heldout-depths --examples-per-depth 128 --log-every 500
python -u train_dynamic_composition.py --config configs/ne_dynamic_300m_depth8_operation_circuit_banks_rank8.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_depth8_operation_circuit_banks_rank8_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_depth8_operation_circuit_banks_rank8_seed18_3000.pt --heldout-depths --examples-per-depth 128 --log-every 500
python -u train_dynamic_composition.py --config configs/ne_dynamic_500m_depth8_operation_circuit_banks_rank8.yaml --steps 3000 --seed 17 --run-id ne_dynamic_500m_depth8_operation_circuit_banks_rank8_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_500m_depth8_operation_circuit_banks_rank8_seed17_3000.pt --heldout-depths --examples-per-depth 128 --log-every 500
python -u train_dynamic_composition.py --config configs/ne_dynamic_500m_depth8_operation_circuit_banks_rank8.yaml --steps 3000 --seed 18 --run-id ne_dynamic_500m_depth8_operation_circuit_banks_rank8_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_500m_depth8_operation_circuit_banks_rank8_seed18_3000.pt --heldout-depths --examples-per-depth 128 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| capacity | train mean | held-out mean | depth 5 | depth 6 | depth 7 | depth 8 | total params | active estimate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 300M, seeds 17/18 | 99.32% | **96.39%** | 98.05% | 96.48% | 95.70% | 95.31% | 7,202,759 | 1,903,600 |
| 500M, seeds 17/18 | 99.32% | 96.29% | 98.05% | 94.92% | 95.70% | **96.48%** | 8,930,759 | 1,903,600 |

The 500M mean is 0.0977 points below 300M. Its active estimate is identical;
only stored factorized capacity grows. The 300M screen already reaches high
accuracy on depths 5--8 after 3,000 steps, so this is a harder task than the
two-step screen but still does not expose a useful 500M advantage.

## Decision

Do not move to 700M or 1B on this architecture. The next capacity attempt
must change the virtual circuit generator itself so additional factor pairs
create new nonlinear functions, not only more addresses over a shared
additive basis. The current 300M operation-bank model remains the reference.
