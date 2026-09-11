# V0.260–V0.261 — rank128 full-range codec long training

**Date:** 2026-09-11  
**Status:** **RETAINED AS PEAK OPT-IN CHECKPOINT**

## Question

V0.255 already reached `98.05%` aggregate held-out accuracy with the
full-range all-operation codec. The remaining errors were concentrated in
mixed-operation programs, while the terminal exact packet was already known
to be correct. This continuation tests whether those errors are simply codec
underfit.

The body, router, circuit bank, codec range, and rank128 head are unchanged;
only the frozen overlay receives 10,000 additional steps, for 15,000 total.

## Protocol

- initialization: V0.255 full-range all-operation overlay
- seeds: `17`, `18`
- total overlay training: `15,000` steps, batch `128`
- train values: `0..95`; held-out depths: `3..4`
- matched evaluation: `256` examples per depth, values `0..95`
- unseen evaluation: fixed operand `96`
- full model: `7,838,217` parameters
- trainable overlay: `337,474` parameters
- body/router/circuit bank: frozen

## Results

### Matched held-out range `0..95`

| Variant | All ops | Depth 3 | Depth 4 | Add | Subtract | Multiply | Mul d3 | Mul d4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0.255, 5k | 98.0469% | 99.4141% | 96.6797% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| **V0.260–261, 15k** | **99.8047%** | **99.8047%** | **99.8047%** | **100.0000%** | **100.0000%** | **99.9023%** | **100.0000%** | **99.8047%** |

Seed17 reached `100%` on every matched operation and depth. Seed18 reached
`99.6094%` all operations and `99.8047%` multiply.

### Unseen fixed operand `96`

| Variant | All ops | Depth 3 | Depth 4 | Add | Subtract | Multiply |
|---|---:|---:|---:|---:|---:|---:|
| V0.255, 5k | 95.3125% | 96.4844% | 94.1406% | 100.0000% | 100.0000% | 100.0000% |
| **V0.260–261, 15k** | **96.6797%** | **95.1172%** | **98.2422%** | **100.0000%** | **100.0000%** | **100.0000%** |

Both seeds remain exactly correct on each individual operation at unseen 96.

## Decision and interpretation

**Retained as the peak opt-in checkpoint.** Extra codec optimization closes
almost all mixed-operation errors without changing the body or router. V0.255
is still the faster 5k-training checkpoint; V0.260–261 is the higher-quality
15k-training checkpoint.

The earlier paired ablation on the same exact-codec architecture found that
turning circuit residual computation off changed matched accuracy by `0.0 pp`
for both seeds. Therefore this line is a successful numeric readout/control,
not evidence that sparse circuits or routing solved the task. The exact
algebraic state supplies the known benchmark semantics.

The next research gate is consequently not 700M/1B scaling. It is to remove
or weaken the hand-coded algebraic prior and measure whether the learned
circuits can reproduce the same value contract on a task whose operation
semantics are not supplied explicitly.

## Artifacts

- config: `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_interaction16_hardcontext_integeroverlay_codec_calibration_w005_all_fullrange.yaml`
- checkpoints: `results/checkpoints/v0_260_rank128_fullrange_codec_continued_seed17_15000.pt`,
  `results/checkpoints/v0_261_rank128_fullrange_codec_continued_seed18_15000.pt`
- taskwise reports: `results/runs/v0_260_rank128_fullrange_codec_continued_taskwise_seed17.json`,
  `results/runs/v0_261_rank128_fullrange_codec_continued_taskwise_seed18.json`
