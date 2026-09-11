# V0.241 — algebraic state write bridge on clean wide-support arithmetic

**Date:** 2026-09-11  
**Status:** `REJECTED FOR MAIN QUALITY ADOPTION`

## Question

V0.240 showed that hard digit context improves subtract but leaves the
multiply ceiling unchanged. This experiment reuses the existing exact
polynomial/Fourier packet at the state-write boundary, testing whether the
learned recurrent state simply loses the semantic value signal between
steps. The bridge adds no new parameters and is enabled only by
`algebraic_state_write_scale=1.0`.

## Protocol

- V0.240 hard-context interaction rank-16 treatment plus the write bridge;
- same V0.239/V0.240 no-interaction/hard-context geometry and circuit bank;
- operands `0..95`, train depths `1..2`, held-out depths `3..4`;
- target offset `134,217,728`, four-digit base-512 compact evaluator;
- 5000 steps, batch `128`, seeds `17` and `18`;
- task-wise held-out comparison against V0.240 hard-context checkpoints.

## Results

Aggregate taskwise held-out metrics are:

| Metric | V0.240 hard context | V0.241 write bridge | Delta |
|---|---:|---:|---:|
| All-operations accuracy | 82.7148% | 83.3008% | +0.5859 pp |
| Depth-3 accuracy | 86.5234% | 87.1094% | +0.5859 pp |
| Depth-4 accuracy | 78.9063% | 79.4922% | +0.5859 pp |
| All-operations CE | 2.682122 | 2.611871 | -0.070251 |

The operation-wise result does not support a multiply solution:

| Operation | V0.240 hard context | V0.241 write bridge | Delta |
|---|---:|---:|---:|
| Add | 100.0000% | 100.0000% | +0.0000 pp |
| Subtract | 99.7070% | 97.4609% | -2.2461 pp |
| Multiply | 15.4297% | 15.4297% | +0.0000 pp |

Multiply depth-3 remains `23.0469%` and depth-4 remains `7.8125%`. The small
aggregate improvement is therefore not evidence that the recurrent multiply
transition was repaired; it is accompanied by a subtract regression.

The model remains at `7,500,743` total and approximately `2,201,584` active
parameters. No capacity increase was used.

## Decision

The pre-writer algebraic bridge is **REJECTED FOR MAIN QUALITY ADOPTION**.
It is not made default and is not a reason to scale to 700M/1B. The result
narrows the hypothesis: merely injecting the exact packet before the learned
writer does not make the learned state compositional for multiply. The next
test changes the read dataflow itself by using the algebraic packet as the
authoritative query-side state, while preserving the same sparse circuits and
clean wide-support protocol.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_writebridge.yaml`
- `results/runs/v0_241_wide_range_writebridge_seed17_5000.json`
- `results/runs/v0_241_wide_range_writebridge_seed18_5000.json`
- `results/runs/v0_241_wide_range_writebridge_taskwise_seed17.json`
- `results/runs/v0_241_wide_range_writebridge_taskwise_seed18.json`
