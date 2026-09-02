# V0.33 role-anchored routing

Status: **positive at 20M, negative at 100M; do not scale this version**

## Change

The router separated the query into two signals. A role-only tree selected a
coarse anchor cell from `operator + stage`; the full numeric register query
then scored candidate circuit rows inside that cell. Unlike V0.32, the bank
was not hard-partitioned by operator.

## Full-domain results

| Model | All nine pairs | Two hidden pairs |
|---|---:|---:|
| 20M role-anchor | 58.03% | **71.13%** |
| 100M role-anchor | 58.69% | 78.32% |
| Existing references | 58.10% / 59.23% | 64.89% / 89.16% |

The 20M hidden result was a real improvement over the 20M reference, but the
100M result failed to reproduce the 100M reference. The same configuration was
used at both sizes.

## Route audit

On the hidden full-domain grid, only 3 of 16 anchors and 264 of 1,408 circuit
rows were used at 20M. At 100M only 4 of 16 anchors and 1,946 of 7,800 rows
were used. The role-only coarse tree therefore collapsed onto too few cells.

## Decision

This is a useful diagnostic, not a scalable solution. It proves that separating
role and value signals can help a small bank, but a learned coarse anchor needs
load balancing or a different bank mechanism before it can scale.
