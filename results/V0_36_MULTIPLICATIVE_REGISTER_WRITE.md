# V0.36 multiplicative register interaction

Status: **strong positive at 20M; capacity scaling still unresolved**

## Change

The typed-register graph was retained, but each ordered operand pair gained an
explicit elementwise-product branch before the routed circuit write:

```text
pair(left,right) = MLP([left,right]) + MLP(left * right)
```

The product branch supplies a direct bilinear interaction for arithmetic
composition. It does not add attention or a Transformer block, and only the
selected circuit rows remain active in the sparse bank.

## 20M all-pairs run

```text
python train_composition.py --config configs/ne_typed_register_20m_multiplicative_all.yaml --steps 10000 --device cuda --run-id ne_typed_register_20m_multiplicative_all_10000 --output results/runs --examples-per-task 512 --log-every 1000 --checkpoint results/checkpoints/ne_typed_register_20m_multiplicative_all_10000.pt
```

Hardware: NVIDIA GeForce RTX 3060 12 GiB. Seed: 17. Training time: 595.6 s.
The run used 19,953,921 total parameters and an estimated 1,684,737 active
parameters (8.44%).

The deterministic `64^3` grid over all nine ordered operator pairs reached
**94.9515%**:

| Pair | Accuracy |
|---|---:|
| add → add | 95.96% |
| add → subtract | 96.18% |
| add → multiply | 98.54% |
| subtract → add | 96.10% |
| subtract → subtract | 96.15% |
| subtract → multiply | 98.57% |
| multiply → add | 88.75% |
| multiply → subtract | 88.55% |
| multiply → multiply | 95.76% |

This is a large improvement over the old 20M all-pairs reference of 58.10%.

## Hidden-stage run

```text
python train_composition.py --config configs/ne_typed_register_20m_multiplicative.yaml --init-checkpoint results/checkpoints/ne_typed_register_20m_multiplicative_all_10000.pt --steps 5000 --device cuda --run-id ne_typed_register_20m_multiplicative_hidden_5000 --output results/runs --examples-per-task 512 --log-every 1000 --checkpoint results/checkpoints/ne_typed_register_20m_multiplicative_hidden_5000.pt
```

The full hidden grid reached **97.27%**: add → multiply 97.90% and multiply →
add 96.64%. Evaluating all nine pairs after hidden adaptation still produced
98.73%, so the improvement was not explained by catastrophic forgetting of the
visible pairs.

## 100M scale control

The same multiplicative pair interaction and global router were trained at
100M for 10,000 steps. The model has 103,410,953 parameters and an estimated
1,687,817 active parameters (1.63%). The deterministic full `64^3` all-pairs
score was **94.13%**:

| Pair | Accuracy |
|---|---:|
| add → add | 95.66% |
| add → subtract | 95.86% |
| add → multiply | 97.85% |
| subtract → add | 94.20% |
| subtract → subtract | 94.47% |
| subtract → multiply | 97.08% |
| multiply → add | 88.50% |
| multiply → subtract | 88.84% |
| multiply → multiply | 94.67% |

The 100M result is 0.82 percentage points below the 20M result. Therefore the
interaction architecture is accepted as the current quality path, but naive
capacity growth is not accepted as a scaling law. The active fraction also
falls from 8.44% to 1.63%, confirming that stored rows grow much faster than
the computation actually used per example.

## Decision

Keep V0.36 as the new reference architecture. Do not spend the next run on a
300M copy with the same route geometry. The next architecture should represent
capacity with reusable/factorized basis components or a scale-invariant route
address, so that larger banks do not fragment credit assignment while keeping
the active path sparse.
