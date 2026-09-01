# V0.30 active-budget audit

V0.30 tests whether the plateau is caused mainly by too few active
micro-circuits per stage.  The typed-register architecture, value-dependent
router, seed, optimizer, training schedule, and deterministic grid are kept
fixed; only `active_circuits` changes from 8 to 16.

| Variant | Active circuits | Active estimate | All-pairs grid | Hidden-pair grid |
|---|---:|---:|---:|---:|
| 20M V0.28 routed | 8 | 1.51M | **57.67%** | **65.56%** |
| 20M active-budget test | 16 | 1.64M | 57.06% | 63.35% |

The active circuit-bank work approximately doubles, but the full active
estimate rises only from 1.54M to 1.64M because the shared encoder/router
dominates the accounting.  The 16-circuit run also takes roughly 1.5× as long
on the local RTX 3060.  It does not improve either deterministic metric and
slightly lowers hidden-pair accuracy.

## Decision

Do not increase active circuits as the next scaling strategy.  The evidence
now points away from raw dormant capacity and raw active width as the primary
limitation.  The next architectural work should target credit assignment and
dataflow supervision: make the intermediate register target causally useful,
measure gradients/routes per composition stage, and test a controlled
operator-function curriculum before another capacity jump.

The experiment remains available in:

```text
configs/ne_typed_register_20m_active16_all.yaml
configs/ne_typed_register_20m_active16.yaml
```

Verification: the repository test suite passes with 39 tests and two expected
Transformer warnings.
