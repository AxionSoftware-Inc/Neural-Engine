# V0.40 depth-capped global routing

Status: **rejected as the 500M scaling fix**

## Change

V0.40 kept the V0.37 factorized virtual circuit bank but reduced the global
hierarchical router from depth 6 to depth 5 at 500M. The purpose was to test
whether the largest bank was losing quality because a deeper tree fragmented
route reuse.

## Result

The model had 38,600 virtual circuits, 19.0M physical parameters, and an
estimated 1.79M active path. After 10,000 all-pairs steps, the deterministic
full `64^3` score was **94.81%**.

| Variant | Full all-pairs |
|---|---:|
| 500M factorized, global depth 6 | 94.72% |
| 500M factorized, global depth 5 | 94.81% |
| 500M factor-address router | 95.55% |
| 300M factorized reference | **96.40%** |

Depth capping gives only a 0.09-point improvement and does not recover the
300M reference. It is not enough to explain or solve the 500M regression.

## Decision

Reject as the default. Keep the 300M factorized global model as the best
current checkpoint. The next validation should focus on second-seed stability,
longer training, and a balanced hidden-composition curriculum rather than more
blind 500M router variants.
