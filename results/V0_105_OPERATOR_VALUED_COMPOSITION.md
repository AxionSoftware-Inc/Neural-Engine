# V0.105 Operator-Valued Layer on Neural Engine Composition

## Question

Does the positive synthetic operator-basis result transfer to the existing
Neural Engine composition benchmark when only one ordinary transform is
replaced?

## Isolation

The current 300M operation-bank model is unchanged except for
`product_encoder[1]`: its ordinary `384 -> 384` linear layer is replaced by
the canonical dense operator-valued layer (`g=16,q=8`). Router, circuit bank,
active route budget, recurrent register graph, data, and training schedule are
unchanged.

```powershell
python -u train_composition.py --config configs/ne_dynamic_300m_composition_operator_valued_product_broad_values.yaml --steps 9000 --seed 17 --run-id ne_dynamic_300m_composition_operator_valued_product_seed17_9000 --output results/runs --checkpoint results/checkpoints/ne_dynamic_300m_composition_operator_valued_product_seed17_9000.pt --examples-per-task 128 --log-every 1000
python -u train_composition.py --config configs/ne_dynamic_300m_composition_operator_valued_product_broad_values.yaml --steps 9000 --seed 18 --run-id ne_dynamic_300m_composition_operator_valued_product_seed18_9000 --output results/runs --checkpoint results/checkpoints/ne_dynamic_300m_composition_operator_valued_product_seed18_9000.pt --examples-per-task 128 --log-every 1000
```

## Results

| model | seed 17 held-out | seed 18 held-out | mean | train mean | total params | active estimate |
|---|---:|---:|---:|---:|---:|---:|
| operator-valued product encoder | 67.19% | 69.53% | **68.36%** | 100.00% | 7,220,615 | 1,718,688 |
| V0.80 operation-bank reference | 80.27% | 78.91% | **79.59%** | 100.00% | 7,361,415 | 1,859,488 |

The operator layer loses 11.23 percentage points on held-out operation order,
despite fitting the training pairs perfectly. This is a real negative result,
not a capacity shortage: the structured synthetic gate passed, but this
placement restricts the feature transform in a way the composition graph needs.

## Decision

Reject this operator-valued placement for the current Neural Engine reference.
Keep the generic operator layer and its positive synthetic evidence in the
branch for future isolated research, but do not combine it with MacroCells or
increase `q` merely to recover the baseline; that would weaken the intended
compression and add a new confound. Move to the next independent architecture
proposal.

## Artifacts

- `neural_engine/operator_valued.py`
- `configs/ne_dynamic_300m_composition_operator_valued_product_broad_values.yaml`
- `results/runs/ne_dynamic_300m_composition_operator_valued_product_seed17_9000.json`
- `results/runs/ne_dynamic_300m_composition_operator_valued_product_seed18_9000.json`
