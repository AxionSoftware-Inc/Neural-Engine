# V0.255 — full-range integer codec calibration

**Date:** 2026-09-11  
**Status:** **RETAINED AS LEADING OPT-IN NUMERIC CANDIDATE**

## Question

V0.254 made the exact integer digit head available to add, subtract, and
multiply and reached nearly 98% held-out accuracy. Its remaining failure on
the unseen fixed value `96` was multiply `0%`, because the calibration range
ended near `81M` while the four-step benchmark can produce much larger legal
values. This screen keeps V0.254 frozen except for another 5,000-step overlay
continuation and calibrates the codec across every legal class target.

With `num_classes=8,589,934,592` and `target_offset=134,217,728`, the sampled
raw codec range is:

```text
[-134,217,728, 8,455,716,863]
```

which maps to target classes `0..8,589,934,591`. The codec calibration weight
remains `0.05`; only the range changes.

## Protocol

- initialization: V0.254 all-operation overlay, one checkpoint per seed
- seeds: `17`, `18`
- overlay training: `5,000` steps, batch `128`
- train values: `0..95`; held-out depths: `3..4`
- matched evaluation: `256` examples per depth, values `0..95`
- unseen-range evaluation: exactly `96` for every operand, same depths
- full model: `7,838,217` parameters
- trainable overlay: `337,474` parameters
- body/router/circuit bank: frozen

## Results

### Matched held-out range `0..95`

| Variant | All ops | Depth 3 | Depth 4 | Add | Subtract | Multiply | Mul d3 | Mul d4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0.252 | 84.1797% | 87.5000% | 80.8594% | 100.0000% | 99.7070% | 59.8633% | 80.0781% | 39.6484% |
| V0.254 | 97.8516% | 98.4375% | 97.2656% | 100.0000% | 100.0000% | 71.8750% | 95.8984% | 47.8516% |
| **V0.255** | **98.0469%** | **99.4141%** | **96.6797%** | **100.0000%** | **100.0000%** | **100.0000%** | **100.0000%** | **100.0000%** |

### Unseen fixed operand `96`

| Variant | All ops | Depth 3 | Depth 4 | Add | Subtract | Multiply |
|---|---:|---:|---:|---:|---:|---:|
| V0.254 | 88.4766% | 89.8438% | 86.7188% | 100.0000% | 100.0000% | 0.0000% |
| **V0.255** | **95.3125%** | **96.4844%** | **94.1406%** | **100.0000%** | **100.0000%** | **100.0000%** |

All table entries are two-seed means. V0.255's matched aggregate was
`98.4375% / 97.6562%` by seed; unseen-96 aggregate was `94.7266% / 95.8984%`.
The unseen multiply result was `100%` in both seeds.

The attempted `96..127` screen was not a valid comparison for this model:
depth-4 targets exceed the configured `8.59B` class range. The next larger
value-range experiment requires either a smaller maximum depth or a larger
classifier class space; it must not silently clip targets.

## Decision and interpretation

**Retained as the leading opt-in quality candidate.** This is the largest
positive movement in the current line: at the matched range it raises
aggregate accuracy from V0.252 `84.1797%` to `98.0469%`, and it fixes unseen-96
multiply from `0%` to `100%`.

The causal conclusion is specific: output-codec range coverage was a major
bottleneck, and the shared router/query was not the only or primary blocker.
This does not validate universal scaling to 700M/1B, and it does not prove
that the learned sparse circuits are solving the arithmetic: the exact
algebraic state supplies the known operation semantics. The default learned
path therefore remains unchanged; V0.255 is an opt-in benchmark candidate.

## Artifacts

- config: `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_integeroverlay_codec_calibration_w005_all_fullrange.yaml`
- checkpoints: `results/checkpoints/v0_255_fullrange_codec_seed17_5000.pt`,
  `...seed18_5000.pt`
- taskwise reports: `results/runs/v0_255_fullrange_codec_taskwise_seed17.json`,
  `...seed18.json`
