# V0.32 stable-family/local-routing prototype

Status: **negative result; do not scale this variant**

## Hypothesis

The proposed family-local router reserved 12.5% of the bank for shared
fallback circuits and routed the remaining circuits inside a stable
operator/stage family. The goal was to stop a larger global bank from
fragmenting route reuse.

## Implementation

The typed-register dataflow graph was preserved. The new router layout was:

```text
[shared fallback][family 0][family 1]...[family 8]
```

The family was supplied deterministically by operator and execution stage.
Within a family, the current numeric query selected local top-k circuits. Half
of the candidate pool came from the local family and half from the shared
fallback block. The active path remained sparse.

No Transformer or attention was added.

## Controlled run

The 20M model used the same seed, 10,000 all-pairs steps, 5,000 hidden-stage
steps, optimizer, batch size and typed-register dataflow as the reference.
The benchmark used the complete `64^3` operand domain.

| Checkpoint | All nine pairs | Two hidden pairs |
|---|---:|---:|
| Existing 20M reference | 58.10% | 64.89% |
| Family-local 20M | 57.76% | 61.37% |

The family-local run took 593.6 seconds for all-pairs training and 300.3
seconds for hidden-stage adaptation. Its active parameter estimate was
1.536M of 19.81M total.

## Decision

This variant is rejected. It did not preserve the all-pairs baseline and lost
3.52 points on the hidden full-domain grid. The likely issue is that the hard
operator/stage family assignment makes each local bank too narrow, while the
shared fallback consumes half of the candidate pool. The result is not evidence
against typed registers; it is evidence against this particular hard family
decomposition.

The experiment was stopped at 20M. It was not scaled to 100M or 300M because a
negative controlled 20M result cannot justify the extra compute.

## Next direction

The next prototype will keep one global physical bank, use a stable learned
role-anchor tree only for coarse route selection, and score the selected local
candidate rows with the full value-dependent query. This avoids both extremes:
full-state global routing and hard private operator partitions.
