# V0.82 Operation Banks with Circuit Input Normalization

## Question

The strongest model still has an asymmetric held-out result, especially when
an add bank receives an accumulator produced by a multiply. This ablation
normalizes the query only on the path entering the sparse circuit bank, with
the intent of reducing composition-dependent scale differences while keeping
the raw query available to the writer and other paths.

## Reproduction

Base source state: `b663954` (`reject per address factor mixing`).

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_input_norm_broad_values.yaml --steps 3000 --seed 17 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_input_norm_broad_values_seed17_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_input_norm_broad_values_seed17_3000.pt --examples-per-task 1024 --log-every 500
python -u train_composition.py --config configs/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_input_norm_broad_values.yaml --steps 3000 --seed 18 --run-id ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_input_norm_broad_values_seed18_3000 --checkpoint results/checkpoints/ne_dynamic_300m_composition_typed_write_adapter_operation_circuit_banks_rank8_input_norm_broad_values_seed18_3000.pt --examples-per-task 1024 --log-every 500
```

Hardware: NVIDIA GeForce RTX 3060, 12,288 MiB, driver 591.86.

## Results

| seed | train | held-out | add -> multiply | multiply -> add | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0000% | 78.1250% | 87.1094% | 69.1406% | 7,362,183 | 1,860,256 |
| 18 | 100.0000% | 78.2227% | 87.9883% | 68.4570% | 7,362,183 | 1,860,256 |
| **mean** | **100.0000%** | **78.1738%** | 87.5488% | 68.7988% | — | — |

The V0.79 rank-8 operation-bank baseline reaches 79.0771% mean at the same
3,000-step budget. Input normalization loses 0.9033 points while adding 768
trainable parameters. It does not improve the hard `multiply -> add` order.

## Decision

Rejected as the default. The circuit bank needs the magnitude information in
the learned query; a generic LayerNorm before the bank is too destructive. Keep
the raw-query circuit path and focus future work on operation-specific
composition interfaces rather than generic scale normalization.

The full test suite passed after implementation: `85 passed, 2 warnings` (the
existing Transformer nested-tensor warnings).
