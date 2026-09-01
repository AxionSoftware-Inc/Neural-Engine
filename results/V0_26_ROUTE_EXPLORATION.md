# V0.26 training-only route exploration

V0.25 showed that simply increasing the circuit bank did not convert stored
capacity into compositional skill. V0.26 tests the `taklif.md` hard-routing
exploration idea: during training only, each tree level has a 5% chance of
choosing a random branch. Evaluation remains deterministic hard routing.

The implementation and configurations were recorded in commit `ecc7a59`
(`Add training-only route exploration`). The 300M exploration run was used as
the parent for the 500M growth run. Both use the hidden operation-order task,
seed 17, 5,000 steps, and LazyAdamW.

## Results

| Run | Total params | Train-split accuracy | Held-out accuracy | Exploration | Train time |
|---|---:|---:|---:|---:|---:|
| NE-300M parent, no exploration | 299.66M | 10.71% | 7.03% | 0% | 331.3 s |
| NE-300M exploration | 299.66M | 11.61% | **11.72%** | 5% | 338.5 s |
| NE-500M parent-growth, no exploration | 505.95M | 8.93% | 9.38% | 0% | 486.6 s |
| NE-500M parent-growth, exploration | 505.95M | 11.83% | **10.94%** | 5% | 491.0 s |
| NE-20M prior reference | 20.37M | 20.09% | 12.50% | 0% | 175.0 s |

Relative to the no-exploration controls, route exploration adds 4.69 held-out
points at 300M and 1.56 points at 500M. The 500M run is still 1.56 points
below the prior 20M reference, and the single-seed differences are not enough
to establish a scaling law. Exploration also increases route randomness during
training, so its effect should be confirmed across seeds before becoming the
default.

## Decision

**WEAK SIGNAL, still NO-GO for 700M/1B.**

Training-only exploration is the first tested intervention that materially
improves the 300M composition result, and the direction survives the 500M
growth step. It does not yet show that larger capacity beats the best smaller
model, so another size jump would hide the real bottleneck rather than solve
it.

Next, keep the same 5% exploration recipe and run a two- or three-seed 300M
confirmation plus a longer composition curriculum. Only if that confirms a
repeatable gain should the recipe be promoted to a 500M quality run. Circuit
family/working-memory changes remain the next alternatives if the seed check
fails.
