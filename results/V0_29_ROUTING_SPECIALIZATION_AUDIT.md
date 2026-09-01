# V0.29 routing specialization and seed audit

V0.29 tests the main failure mode identified after the typed-register pivot:
the router can see enough value information to fragment a large circuit bank
into operand-specific routes, but removing value information entirely can
make the circuit family too rigid.  The experiments stay non-Transformer and
use the same explicit register graph as V0.28:

```text
(a, b, op1) -> partial -> (partial, c, op2) -> final -> readout
```

The primary metric is the deterministic `16^3 = 4,096` operand grid per
operation pair.  The two hidden pairs are `add_then_multiply` and
`multiply_then_add`; all reported hidden scores below are the mean over those
8,192 examples.

## Changes

Three routing controls were added without changing the circuit computation:

- `typed_route_partitions`: restrict each primitive operator and readout to a
  separate bank window;
- `typed_route_shared`: an exploratory shared-plus-private bank layout;
- `route_query_mode: compressed`: route from operator/stage embeddings plus a
  learned 32- or 4-dimensional value bottleneck, while the selected circuit
  still receives the full numeric query.

The router also supports multiple per-example candidate windows.  The default
`value_and_type` behavior is unchanged.

## Results

| Variant | Stored params | All-pairs grid | Hidden-pair grid | Decision |
|---|---:|---:|---:|---|
| 20M routed V0.28 reference | 19.78M | 57.67% | 65.56% | reference |
| 20M partitioned, seed 17 | 19.81M | 56.01% | **87.01%** | promising single seed |
| 20M partitioned, seed 18 | 19.81M | 55.23% | 61.46% | high-variance; not robust |
| 100M partitioned, seed 17 | 103.24M | 56.53% | 68.65% | reject as default |
| 20M shared + private | 19.78M | 56.49% | 60.82% | reject |
| 20M typed-only route | 19.78M | 53.15% | not run | too rigid |
| 20M compressed route, 32-d | 19.81M | **57.92%** | **71.75%** | best new 20M control |
| 100M compressed route, 32-d | 103.26M | 57.16% | 61.22% | no scale conversion |
| 20M compressed route, 4-d | 19.78M | 57.09% | 60.68% | too narrow |

The earlier V0.28 100M routed seed-17 hidden result remains 89.87%, while
the independent seed-18 repeat was 73.79%.  Together with the new
partitioned seed pair, this shows that the large hidden-grid jump is not yet a
stable scaling law.

The direct-readout ablation remains negative: removing the third routed
readout from the V0.28 100M checkpoint reduced hidden grid accuracy to 83.37%
from 89.87%.  The readout route should therefore remain enabled for now.

## Route audit

The route census explains why capacity does not improve monotonically.  On
the 100M partitioned all-pairs checkpoint, stage-3 readout used 1,944 of its
1,950-row window, with within-route Jaccard only 0.004.  The hidden checkpoint
used 1,931 rows and left 32.8% of the full bank dead.  The router is therefore
not collapsed, but its selected circuit sets are highly input-specific and
have little reuse.

The 20M compressed 32-d hidden checkpoint is more local: stage-1 unions are
63 and 79 circuits for the two hidden operator groups, stage-2 unions are 100
and 101, and within-group Jaccard reaches 0.25–0.68.  At 100M the same
32-dimensional context expands to stage unions of 757/401 and 1,607/1,056,
so the bottleneck is still too expressive relative to the larger bank.

## Decision

Do not jump to 300M, 500M, 700M, or 1B based on these routing variants.
Partitioning has a positive single-seed result but fails the second-seed
repeat.  Typed-only routing is too restrictive, while a 32-dimensional
compressed route helps at 20M but does not convert at 100M.

The current best defensible model is still the V0.28 100M routed checkpoint,
reported with its seed variance rather than as a guaranteed 89.87% result.
The next research target should be a controlled active-budget/credit-
assignment experiment: keep value-dependent routing, but increase or train
the active circuit budget gradually and measure whether the additional active
rows, rather than dormant rows alone, improve the deterministic grid.

## Verification

```powershell
$env:PYTHONPATH = (Get-Location).Path
pytest -q
```

The full test suite passes with 39 tests and two expected Transformer
warnings.  Training checkpoints and JSON run reports are kept under the
ignored `results/checkpoints/` and `results/runs/` directories.
