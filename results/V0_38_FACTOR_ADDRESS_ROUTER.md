# V0.38 factor-address router

Status: **improves 500M over global factor routing; not enough to recover 300M**

## Change

The factorized bank from V0.37 was kept, but the router stopped scoring one
independent key per virtual circuit. It scores reusable factor keys, builds a
small Cartesian product of the top factor IDs, and selects virtual circuit
combinations from that product.

## Run and result

The 500M virtual model used 38,600 virtual circuits, factor count 197, 4.21M
physical parameters, and an estimated 1.77M active parameters. After 10,000
all-pairs training steps, the deterministic `64^3` full-domain score was
**95.55%**.

| Pair | Accuracy |
|---|---:|
| add → add | 98.94% |
| add → subtract | 98.79% |
| add → multiply | 98.54% |
| subtract → add | 97.92% |
| subtract → subtract | 97.69% |
| subtract → multiply | 97.94% |
| multiply → add | 88.24% |
| multiply → subtract | 88.29% |
| multiply → multiply | 93.62% |

This improves the same 500M factorized bank with a global router from 94.72%
to 95.55%, but remains below the 300M factorized global result of 96.40%.
The multiply-first compositions remain the bottleneck.

## Decision

Keep the idea as an optional route implementation, but do not make it the
default. The next candidate must preserve factor-pair interaction while
avoiding the top-factor shortlist bottleneck.
