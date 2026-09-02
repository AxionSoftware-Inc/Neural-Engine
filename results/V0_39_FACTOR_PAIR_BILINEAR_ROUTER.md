# V0.39 factor-pair bilinear router

Status: **rejected; wider factor search did not improve the full grid**

## Change

V0.39 extended V0.38 with a 12-factor candidate pool instead of 6 and added a
query-conditioned bilinear score for each factor pair:

```text
score(a,b|q) = score(a|q) + score(b|q)
               + q · (key_a * key_b)
```

The circuit bank and multiplicative register interaction were unchanged.

## Result

The 500M virtual model used the same 38,600 addresses and 4.21M physical
parameters. The deterministic `64^3` all-pairs score after 10,000 steps was
**95.43%**, slightly below V0.38’s 95.55%.

| Pair | Accuracy |
|---|---:|
| add → add | 98.95% |
| add → subtract | 98.85% |
| add → multiply | 97.47% |
| subtract → add | 98.60% |
| subtract → subtract | 98.44% |
| subtract → multiply | 97.24% |
| multiply → add | 88.12% |
| multiply → subtract | 88.20% |
| multiply → multiply | 93.05% |

The wider shortlist and bilinear route score did not fix multiply-first
composition and slightly reduced overall quality.

## Decision

Reject as the default route. V0.37 factorized global routing remains the best
scaling reference; V0.38 is retained as an optional diagnostic variant.
