# V0.103 Structured / Operator-Valued Parameter Gate

## Scope

This is the isolated first gate for `taklif21`. MacroCells, router changes,
memory changes, and factorized circuit changes are all off. The new
`OperatorValuedLinear` layer represents each packet block as

`Theta[o,i] = sum_a coeff[o,i,a] * basis[a]`.

The canonical setting is feature width 384, packet width `g=16`, and eight
dense shared operators `q=8`.

## Correctness and oracle protocol

The implementation test checks that the basis-first forward pass is exactly
equal to an ordinary `torch.nn.functional.linear` call using the materialized
effective matrix. A standard-matrix basis test checks that `q=g^2` can
represent arbitrary packet blocks.

The synthetic experiment uses 24x24 packet blocks. It reports the optimal
rank-`q` SVD approximation and fits learned-basis and fixed-random-basis
students for 1,000 steps on CUDA.

```powershell
python -m pytest -q tests/test_operator_valued.py
python -u experiment_operator_valued.py --device cuda --steps 1000 --output results/runs/operator_valued_synthetic.json
```

## Results

### Implementation gate

`3 passed`. The arbitrary-block standard basis and materialized-forward
checks pass at `1e-6` tolerance.

### Representation and optimization gate

| target/control | method | relative Frobenius error |
|---|---|---:|
| known shared rank-8 target | SVD oracle, q=8 | 0.000035 |
| known shared rank-8 target | learned basis + coefficients | **0.0000048** |
| known shared rank-8 target | fixed random basis + coefficients | 0.9870 |
| random unconstrained target | SVD oracle, q=8 | 0.9593 |
| random unconstrained target | learned basis + coefficients | 0.9593 |
| random unconstrained target | fixed random basis + coefficients | 0.9836 |

The structured target is recovered to numerical precision, while the random
target keeps the expected low-rank/shared-span approximation ceiling. Learned
bases beat the fixed-random control on the structured target.

## Scalar DOF and compute accounting

For the 384x384, `g=16,q=8` layer without bias:

- full dense matrix: 147,456 trainable scalars;
- operator-valued layer: 6,656 stored/trainable scalars;
- effective matrix entries: 147,456;
- parameter ratio: 22.15x fewer stored scalar values;
- basis-first theoretical MAC estimate: 122,880.

The ratio is not a whole-model compression claim and no runtime claim is made
from this arithmetic alone.

## Decision

Gate 0 (implementation correctness) and Gate 1 (representable structured
target) pass. This direction remains alive for a matched synthetic task
comparison. The random-target ceiling is a useful warning: the layer only
helps when repeated operator structure actually exists. Do not combine it with
MacroCells or current circuit routing until the isolated task gate is frozen.

## Artifacts

- `neural_engine/operator_valued.py`
- `tests/test_operator_valued.py`
- `experiment_operator_valued.py`
- `results/runs/operator_valued_synthetic.json`
