# V0.257–V0.258 — rank64 long-training continuation

**Date:** 2026-09-11  
**Status:** **RETAINED AS LOWER-BUDGET OPT-IN; NOT PEAK QUALITY**

## Question

The first rank64 screen (V0.256, 5,000 steps) underfit: subtract was only
`81.05%`. Its training loss was still `0.098`, compared with `0.007` for the
rank128 V0.255 screen. This continuation tests whether rank64 is actually too
small or simply needs more optimization.

The body, router, circuit bank, full-range codec, and task protocol remain
unchanged. Only the rank64 all-operation output head is trained for 15,000
steps from the V0.252-compatible initialization.

## Protocol

- rank64 output head: `207,362` trainable parameters
- full model: `7,708,105` parameters
- seeds: `17`, `18`
- steps: `15,000` each, batch `128`
- train values: `0..95`; held-out depths: `3..4`
- matched evaluation: `256` examples per depth, values `0..95`
- unseen evaluation: fixed operand `96`
- codec calibration: full legal target range, weight `0.05`

## Results

### Matched held-out range `0..95`

| Variant | All ops | Depth 3 | Depth 4 | Add | Subtract | Multiply | Mul d3 | Mul d4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0.255 rank128 | 98.0469% | 99.4141% | 96.6797% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| **V0.257–258 rank64** | **96.9727%** | **98.6328%** | **95.3125%** | **100.0000%** | **99.2188%** | **99.4141%** | **99.2188%** | **99.6094%** |

### Unseen fixed operand `96`

| Variant | All ops | Depth 3 | Depth 4 | Add | Subtract | Multiply |
|---|---:|---:|---:|---:|---:|---:|
| V0.255 rank128 | 95.3125% | 96.4844% | 94.1406% | 100.0000% | 100.0000% | 100.0000% |
| **V0.257–258 rank64** | **95.6055%** | **97.2656%** | **93.9453%** | **100.0000%** | **100.0000%** | **100.0000%** |

The rank64 matched values are means over the two seeds. Seed-wise aggregate
was `96.4844% / 97.4609%`; seed-wise multiply was `99.6094% / 99.2188%`.
Unseen multiply was `100%` for both seeds.

## Decision

**Retain rank64 as a lower-budget opt-in, not as the peak-quality default.**
Longer training fixes the V0.256 underfit, so rank64 is not fundamentally
invalid. It remains about `38.6%` smaller in trainable overlay parameters than
rank128, but loses about `1.07 pp` aggregate and `0.78 pp` depth-3 quality on
the matched task, while taking substantially longer to train.

The practical choices are now explicit:

- rank128 V0.255: best quality and faster convergence;
- rank64 V0.257–258: lower overlay budget, more training, near-perfect quality.

Rank32 is not scheduled: the rank64 result already shows the budget/quality
frontier, and the old rank32 screen failed badly before full-range calibration.

## Artifacts

- config: `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank64_interaction16_hardcontext_integeroverlay_codec_calibration_w005_all_fullrange.yaml`
- checkpoints: `results/checkpoints/v0_257_rank64_fullrange_codec_continued_seed17_15000.pt`,
  `results/checkpoints/v0_258_rank64_fullrange_codec_seed18_15000.pt`
- taskwise reports: `results/runs/v0_257_rank64_fullrange_codec_continued_taskwise_seed17.json`,
  `results/runs/v0_258_rank64_fullrange_codec_taskwise_seed18.json`
