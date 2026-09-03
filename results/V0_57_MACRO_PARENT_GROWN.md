# V0.57 Macro-Cell Parent-Grown Expansion

## Question

The V0.56 scratch Macro-Cell screen showed a positive but non-monotonic result:
adding 256 macro cells did not reliably improve the 20M model. This experiment
tests whether the problem is optimization and route credit assignment rather
than the Macro-Cell capacity itself.

The procedure is:

1. train a 16-cell parent;
2. copy shared weights and the learned parent rows into a 64-cell model;
3. expose the new router levels with unbiased biases and retain their random
   projections;
4. train the 64-cell intermediate model;
5. repeat the same expansion from 64 to 256 cells.

The model remains attention-free. Expansion adds reusable serial local
Macro-Cells and sparse top-1 routing; it does not add a Transformer or a dense
global operation.

## Results

All runs use the dynamic composition task, train on program depths 1--4, and
evaluate on unseen depths 5--6 with 1,024 examples per depth. The staged runs
use 3,000 optimizer steps at each 64- and 256-cell stage. The parent was also
trained for 3,000 steps before expansion.

| Seed | Stage | Params | Train accuracy | Unseen-depth accuracy | Macro route coverage |
|---:|---|---:|---:|---:|---:|
| 17 | 64 cells | 3,443,917 | 98.88% | 97.75% | 64/64 |
| 17 | 256 cells | 8,532,689 | 99.93% | 99.51% | 255/256 |
| 18 | 64 cells | 3,443,917 | 99.61% | 98.68% | 64/64 |
| 18 | 256 cells | 8,532,689 | 99.88% | 99.71% | 254/256 |

The 256-cell staged mean is **99.61%** across the two seeds, and the route
audit shows that the larger bank is actually reachable. This is a strong
positive signal for staged growth and directly addresses the earlier capacity
failure mode.

The decisive equal-step control trains the 256-cell model from scratch for
9,000 steps:

| Seed | Training schedule | Unseen-depth accuracy |
|---:|---|---:|
| 17 | scratch, 9,000 steps | 99.56% |
| 18 | scratch, 9,000 steps | 99.80% |

The scratch mean is **99.68%**, effectively the same as the staged mean within
this two-seed screen (staged minus scratch: **-0.07 percentage points**). The
earlier 3,000-step scratch runs scored only 78.56% and 75.29% (76.93% mean),
so the apparent capacity failure was largely an optimization-budget and
initial-routing problem. Staged growth reaches high quality reliably, but it
does not yet beat a sufficiently trained 256-cell model from scratch.

## Implementation

- `neural_engine/macro_growth.py` copies compatible shared weights and parent
  Macro-Cell rows into a larger bank.
- `train_dynamic_macro_growth.py` provides a direct parent-to-target control.
- `train_dynamic_macro_staged_growth.py` runs the 16 -> 64 -> 256 schedule and
  records per-stage accuracy, parameter counts, and route coverage.
- The expansion behavior is covered by a unit test in
  `tests/test_dynamic_register.py`.

## Decision

Keep staged parent-grown expansion as a useful curriculum and initialization
option, but do not claim a capacity advantage from it. The next gate is a
longer-depth and compositional-generalization test. Only if that gate passes
should the same growth mechanism be taken to a 300M or larger bank; scaling to
700M/1B is not justified by this arithmetic screen alone.
