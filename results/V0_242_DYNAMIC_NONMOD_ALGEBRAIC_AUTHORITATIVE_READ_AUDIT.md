# V0.242 — algebraic authoritative read on clean wide-support arithmetic

**Date:** 2026-09-11  
**Status:** `REJECTED FOR MAIN QUALITY ADOPTION`

## Question

V0.241 showed that adding the exact algebraic packet as a pre-writer hint
does not change multiply. V0.242 makes the same packet the authoritative
representation read by the pair encoder and router. The learned accumulator
continues to receive the normal sparse circuit/state-writer update; only the
query-side read dataflow changes.

## Protocol

- V0.240 hard-context rank-16 interaction and the same factorized circuit bank;
- `algebraic_state_authoritative_read=true`, polynomial2/Fourier packet;
- operands `0..95`, train depths `1..2`, held-out depths `3..4`;
- target offset `134,217,728`, four-digit base-512 compact evaluator;
- 5000 steps, batch `128`, seeds `17` and `18`;
- task-wise held-out comparison against V0.240 hard-context.

## Results

| Metric | V0.240 hard context | V0.242 authoritative read | Delta |
|---|---:|---:|---:|
| All-operations accuracy | 82.7148% | 82.2266% | -0.4883 pp |
| Depth-3 accuracy | 86.5234% | 86.7188% | +0.1953 pp |
| Depth-4 accuracy | 78.9063% | 77.7344% | -1.1719 pp |
| All-operations CE | 2.682122 | 2.608950 | -0.073172 |

Task-wise held-out accuracy is:

| Operation | V0.240 hard context | V0.242 authoritative read | Delta |
|---|---:|---:|---:|
| Add | 100.0000% | 99.9023% | -0.0977 pp |
| Subtract | 99.7070% | 95.6055% | -4.1016 pp |
| Multiply | 15.4297% | 15.4297% | +0.0000 pp |

Multiply depth-3 remains `23.0469%` and depth-4 remains `7.8125%`.
Therefore removing the learned accumulator from the query read does not
repair multiply; it slightly damages the already strong subtract path.

The model remains at `7,500,743` total and approximately `2,201,584` active
parameters. No capacity increase was used.

## Decision

The authoritative-read dataflow is **REJECTED FOR MAIN QUALITY ADOPTION**.
The exact packet is informative, but neither additive write exposure nor
query-side replacement is sufficient. The remaining bottleneck is likely the
final value-to-digit codec or a missing compositional output representation,
not just the router's choice of state source. The next test should isolate a
learned digit decoder fed directly by the algebraic packet, while preserving
the sparse recurrent body as a control; no 700M/1B scaling follows.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_authoritative_read.yaml`
- `results/runs/v0_242_wide_range_authoritative_read_seed17_5000.json`
- `results/runs/v0_242_wide_range_authoritative_read_seed18_5000.json`
- `results/runs/v0_242_wide_range_authoritative_read_taskwise_seed17.json`
- `results/runs/v0_242_wide_range_authoritative_read_taskwise_seed18.json`
