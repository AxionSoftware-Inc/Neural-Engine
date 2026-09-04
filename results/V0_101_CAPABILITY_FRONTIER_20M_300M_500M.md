# V0.101 Capability Frontier: 20M, 300M, 500M

## Question

Does the operation-specific factorized architecture benefit from additional
virtual capacity on the harder compositional task, or was the earlier
300M/500M plateau caused by a saturated benchmark?

## Protocol

All runs train on program depths 1--4 and evaluate on unseen depths 5--8.
They use 3,000 steps, batch size 512, 128 examples per held-out depth, two
seeds (17 and 18), the same value range 0--63, and CUDA on the RTX 3060.
The 20M run below uses the same operation-specific circuit-bank and rank-8
write-adapter family as the 300M/500M reference; its virtual bank has 1,408
addresses and 38 reusable factor rows. The older `ne_dynamic_20m_depth8.yaml`
run is retained as a non-matched baseline and is not used for the scaling
claim.

```powershell
python -u train_dynamic_composition.py --config configs/ne_dynamic_20m_depth8_operation_circuit_banks_rank8.yaml --steps 3000 --seed 17 --run-id ne_dynamic_20m_depth8_operation_circuit_banks_rank8_seed17_3000 --output results/runs --checkpoint results/checkpoints/ne_dynamic_20m_depth8_operation_circuit_banks_rank8_seed17_3000.pt --heldout-depths --examples-per-depth 128 --log-every 500
python -u train_dynamic_composition.py --config configs/ne_dynamic_20m_depth8_operation_circuit_banks_rank8.yaml --steps 3000 --seed 18 --run-id ne_dynamic_20m_depth8_operation_circuit_banks_rank8_seed18_3000 --output results/runs --checkpoint results/checkpoints/ne_dynamic_20m_depth8_operation_circuit_banks_rank8_seed18_3000.pt --heldout-depths --examples-per-depth 128 --log-every 500
```

The 300M and 500M rows reuse the completed V0.96 runs under the identical
depth protocol.

## Results

| virtual capacity | train mean | held-out mean | depth 5 | depth 6 | depth 7 | depth 8 | total params | active estimate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20M, seeds 17/18 | 96.09% | 90.04% | 92.19% | 90.23% | 88.67% | 89.06% | 2,748,359 | 1,903,600 |
| 300M, seeds 17/18 | 99.32% | **96.39%** | 98.05% | 96.48% | 95.70% | 95.31% | 7,202,759 | 1,903,600 |
| 500M, seeds 17/18 | 99.32% | 96.29% | 98.05% | 94.92% | 95.70% | **96.48%** | 8,930,759 | 1,903,600 |

The matched 20M runs are 6.35 points below 300M, so virtual capacity does
help before the plateau. Increasing 300M to 500M changes only stored factor
capacity while the active estimate stays fixed and gives -0.10 points. The
current evidence does not justify a 700M/1B run on this exact benchmark.

## Decision

Keep the 300M operation-bank model as the quality reference. Do not spend the
next run on another scale jump. The remaining question is whether the sparse
execution itself is wasting the available capacity; test grouped/contiguous
execution and locality next, independently of model quality.

## Run artifacts

- `results/runs/ne_dynamic_20m_depth8_operation_circuit_banks_rank8_seed17_3000.json`
- `results/runs/ne_dynamic_20m_depth8_operation_circuit_banks_rank8_seed18_3000.json`
- `results/V0_96_DYNAMIC_DEPTH_CAPACITY_SCREEN.md`
