# V0.248 — frozen multiply integer overlay, rank 32

**Date:** 2026-09-11  
**Status:** `REJECTED FOR QUALITY`

V0.248 halves the multiply overlay projection rank from 64 to 32 while
keeping the V0.240 base head at rank 128. The base checkpoint is frozen and
only the exact integer decoder and multiply-specific output head are trained.

| Held-out metric | V0.240 | V0.248 rank32 | Delta |
|---|---:|---:|---:|
| All operations | 82.7148% | 78.4180% | −4.2969 pp |
| Depth 3, all operations | 86.5234% | 81.4453% | −5.0781 pp |
| Depth 4, all operations | 78.9063% | 75.3906% | −3.5156 pp |
| Add | 100.0000% | 100.0000% | +0.0000 pp |
| Subtract | 99.7070% | 99.7070% | +0.0000 pp |
| Multiply | 15.4297% | 13.6719% | −1.7578 pp |
| Multiply, depth 3 | 23.0469% | 19.7266% | −3.3203 pp |
| Multiply, depth 4 | 7.8125% | 7.6172% | −0.1953 pp |

The frozen paths are preserved, but rank32 cannot learn a useful multiply
codec: the full model has `7,643,049` parameters and only `142,306` trainable
overlay parameters. Rank32 is rejected; rank128 remains the quality choice,
with rank64 retained only as a budget diagnostic.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_integeroverlay_rank32.yaml`
- `results/runs/v0_248_frozen_integeroverlay_rank32_taskwise_seed17.json`
- `results/runs/v0_248_frozen_integeroverlay_rank32_taskwise_seed18.json`
