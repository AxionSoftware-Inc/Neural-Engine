# V0.52 dynamic capacity and routing follow-up

Status: **shared factor mix is promoted as the current bank design; capacity
scaling and value-independent routing are not accepted as quality fixes**

## Protocol

The main capacity screen uses the attention-free Dynamic Register Machine with
shared factor mixing, 3,000 optimizer steps, batch size 512, seed 17, and
training on program depths 1–4. Evaluation uses unseen depths 5–6 with the
same full value range 0–63. The earlier per-address-mix 20M/100M/300M results
remain the comparison baseline.

The routing diagnostic uses the stricter OOD gate: train on values 0–31 and
depths 1–4, evaluate on values 32–63 and depths 5–6. It changes no labels or
active circuit budget.

## Capacity screen with shared factor mixing

| Model | Physical parameters | OOD depth 5–6 | Train accuracy | Time |
|---|---:|---:|---:|---:|
| 300M, shared mix, seed 17 | 3.26M | 83.20% | 91.89% | 341s |
| 300M, shared mix, seed 18 | 3.26M | 83.11% | 91.50% | 343s |
| 500M, shared mix, seed 17 | 3.83M | 83.01% | 91.06% | 346s |
| 700M, shared mix, seed 17 | 4.31M | 80.76% | 89.01% | 349s |

The two-seed 300M shared-mix mean is **83.16%**, versus **78.62%** for the
old per-address-mix 300M mean: **+4.54 percentage points**. The improvement
is repeatable across two seeds. It is most plausibly a regularization and
parameterization benefit: a single shared two-coefficient factor mix removes
one value-dependent table of mix weights from every virtual address.

Increasing the virtual bank from 300M to 500M did not improve the holdout, and
700M regressed. Therefore raw virtual capacity is not the current bottleneck.
These are scratch-trained models, not parent-grown checkpoints.

The strict unseen-value gate remains negative: 20M shared mix reaches 34.57%
and 300M shared mix reaches 34.77%, while both fit the training range at about
99.7%. Shared mixing improves the depth holdout but does not solve
value-independent add/subtract extrapolation.

## Scale stress

The shared mix removes the linear per-address mix tensor. In the forward-only
stress on the local GPU, active parameters stayed at about 1.45M while the
virtual bank grew from 100k to 1M. Measured latency stayed around 29–33ms per
batch of 256, and peak memory rose from about 112MB to 149MB. This validates
the storage direction, not an 8B training claim; an 8B physical run has not
been instantiated.

## Value-independent routing diagnostic

`route_context_mode=operation_step` gives the router only operation and step
embeddings. Circuit execution still receives the full value-dependent query.
This directly tests whether unseen-value failure is caused only by route
fragmentation.

| Variant | OOD accuracy | Train accuracy | Observed virtual routes | Decision |
|---|---:|---:|---:|---|
| Full query routing baseline | 34.77% | 99.76% | value-dependent reuse | reference |
| Operation/step-only routing | 31.25% | 99.51% | 24 stable routes | reject |

The stable route pattern did not recover the missing arithmetic. The failure
is therefore deeper than router drift: the recurrent register and circuit
composition do not carry a sufficiently strong modular arithmetic prior.

## Decision and next experiment

Keep shared factor mixing as the current scalable bank implementation. Do not
move to 1B or 8B on capacity alone. The next architecture test should add a
modular composition prior to the register/circuit interface (or run an exact
modular control first), then require the same unseen-value gate to improve
before further scale spending.

## Reproduction run IDs

```text
ne_dynamic_300m_shared_mix_depth_holdout_3000
ne_dynamic_300m_shared_mix_depth_holdout_seed18_3000
ne_dynamic_300m_shared_mix_unseen_values_3000
ne_dynamic_500m_shared_mix_depth_holdout_3000
ne_dynamic_700m_shared_mix_depth_holdout_3000
ne_dynamic_20m_route_operation_step_unseen_values_3000
```
