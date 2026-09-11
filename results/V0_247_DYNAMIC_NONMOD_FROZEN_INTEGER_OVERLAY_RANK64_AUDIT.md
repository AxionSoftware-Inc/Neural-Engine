# V0.247 — frozen multiply integer overlay, rank 64

**Date:** 2026-09-11  
**Status:** `RETAINED AS LOWER-BUDGET DIAGNOSTIC; RANK128 REMAINS LEADING`

## Question

V0.246 showed that a frozen exact-integer decoder and a separate
multiply-specific digit head improve multiply without changing add/subtract.
Can the overlay head use half the projection rank while retaining most of the
gain? The base V0.240 output head stays rank 128; only the new multiply head is
rank 64.

## Results

| Held-out metric | V0.240 | V0.246 rank128 | V0.247 rank64 |
|---|---:|---:|---:|
| All operations | 82.7148% | 83.6914% | 82.8125% |
| Depth 3, all operations | 86.5234% | 87.3047% | 85.9375% |
| Depth 4, all operations | 78.9063% | 80.0781% | 79.2969% |
| Add | 100.0000% | 100.0000% | 100.0000% |
| Subtract | 99.7070% | 99.7070% | 99.7070% |
| Multiply | 15.4297% | 23.1445% | 19.9219% |
| Multiply, depth 3 | 23.0469% | 36.9141% | 31.2500% |
| Multiply, depth 4 | 7.8125% | 9.3750% | 8.5938% |

The rank64 result is positive and preserves the frozen operations, but it keeps
only about 58% of the rank128 multiply gain (`+4.4922 pp` versus `+7.7148
pp`) and almost none of the aggregate gain (`+0.0977 pp`). Its full model has
`7,708,105` parameters and only `207,362` trainable overlay parameters,
compared with `7,838,217` and `337,474` for rank128.

## Decision

Rank64 is retained as a lower-budget diagnostic, not selected over rank128.
The result confirms a quality/budget tradeoff rather than a new architecture
ceiling. The next screen is rank32, after which the overlay rank can be fixed
and the work can return to the more fundamental recurrent dataflow problem.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_integeroverlay_rank64.yaml`
- `results/runs/v0_247_frozen_integeroverlay_rank64_seed17_5000.json`
- `results/runs/v0_247_frozen_integeroverlay_rank64_seed18_5000.json`
- `results/runs/v0_247_frozen_integeroverlay_rank64_taskwise_seed17.json`
- `results/runs/v0_247_frozen_integeroverlay_rank64_taskwise_seed18.json`
