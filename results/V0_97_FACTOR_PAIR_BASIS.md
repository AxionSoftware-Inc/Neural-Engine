# V0.97 Nonlinear Factor-Pair Basis

## Question

The V0.96 depth screen did not produce a useful 500M advantage, so this
experiment changes the virtual circuit generator rather than only increasing
the number of factor addresses. Each selected factor pair gets a learned
nonlinear code. The code is combined with shared down/up/bias basis tensors,
so pair interactions can express functions that are not merely additive
combinations of two factor rows.

## Reproduction

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_factor_pair_basis_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_factor_pair_basis_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_factor_pair_basis_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_factor_pair_basis_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_factor_pair_basis_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_factor_pair_basis_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86. The result
JSON files are preserved under `results/runs/`.

## Results

| variant | seed 17 held-out | seed 18 held-out | two-seed mean | add -> multiply | multiply -> add | total params | active estimate |
|---|---:|---:|---:|---:|---:|---:|---:|
| factor-pair basis | 79.39% | 78.81% | 79.10% | 88.77% | 69.43% | 7,669,239 | 2,062,752 |
| V0.80 shared factor baseline | 80.27% | 78.91% | **79.59%** | **90.58%** | 68.60% | 7,361,415 | 1,859,488 |

Training accuracy was 100% for both seeds. The pair basis improves the harder
`multiply -> add` direction by about 0.83 points, but loses about 1.81 points
on `add -> multiply`; the net held-out mean is 0.49 points worse. It also
increases the active estimate by about 203K parameters without improving the
overall result.

## Decision

Reject nonlinear factor-pair basis as the default architecture. It confirms
that adding a shared pair correction is not enough to make capacity scale:
the model can fit the training compositions, but the unseen-order transfer
remains asymmetric and does not improve reliably across seeds. Do not spend
the next budget on 500M/700M/1B versions of this exact pair-basis design.

The current reference remains the V0.80 300M operation-bank model. Any next
architecture should address the asymmetric composition interface directly,
with a fixed two-seed held-out gate and the same training budget before a
larger capacity run.
