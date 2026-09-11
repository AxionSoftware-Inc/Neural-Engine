# V0.249–V0.252 — exact integer codec calibration sweep

**Date:** 2026-09-11  
**Status:** `V0.252 RETAINED AS LEADING OPT-IN; BASE DEFAULT UNCHANGED`

## Question

The frozen integer overlay improves multiply, but the original overlay sees
mostly shallow values during training. This sweep adds a synthetic codec loss:
random raw integer values in `[-81,450,625, 81,450,625]` are encoded through
the same exact integer digit embeddings and decoder, then supervised against
the four base-512 digits of `target_offset + value`. The model body and the
learned add/subtract path remain frozen. Only the codec and multiply head are
updated.

The range is the maximum magnitude of a four-operand `0..95` product. The
calibration loss weight is the only changed variable across V0.249–V0.252.

## Matched taskwise results

| Variant | Calibration weight | All ops | Depth 3 | Depth 4 | Add | Subtract | Multiply | Mul d3 | Mul d4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0.240 base | — | 82.7148% | 86.5234% | 78.9063% | 100.0000% | 99.7070% | 15.4297% | 23.0469% | 7.8125% |
| V0.246 overlay | 0 | 83.6914% | 87.3047% | 80.0781% | 100.0000% | 99.7070% | 23.1445% | 36.9141% | 9.3750% |
| V0.249 | 1.00 | 77.6367% | 81.8359% | 73.4375% | 100.0000% | 99.7070% | 38.3789% | 51.5625% | 25.1953% |
| V0.250 | 0.25 | 81.4453% | 85.9375% | 79.6875% | 100.0000% | 99.7070% | 46.8750% | 62.1094% | 31.6406% |
| V0.251 | 0.10 | 82.8125% | 85.9375% | 79.6875% | 100.0000% | 99.7070% | 52.9297% | 69.9219% | 35.9375% |
| **V0.252** | **0.05** | **84.1797%** | **87.5000%** | **80.8594%** | **100.0000%** | **99.7070%** | **59.8633%** | **80.0781%** | **39.6484%** |

All means are over seed17/18 with `256` examples per depth and the same
held-out depth protocol as V0.240. V0.252 improves aggregate accuracy by
`+1.4648 pp` and multiply by `+44.4336 pp` while preserving add/subtract.
Its seed-wise held-out multiply is `60.9375%/58.7891%`, so the direction is
consistent across seeds. The full model remains `7,838,217` parameters, but
only `337,474` are trainable in the overlay.

## Decisions

- V0.249 weight1.0: **rejected**; deep multiply improves but aggregate and
  shallow multiply regress too much.
- V0.250 weight0.25: **diagnostic only**; deep gain is strong but aggregate is
  below baseline.
- V0.251 weight0.10: **retained diagnostic**; aggregate is nearly neutral and
  deep multiply improves strongly.
- V0.252 weight0.05: **leading opt-in candidate**; best aggregate and deep
  multiply result in this sweep. It is not made the default architecture
  because it is an operation-specific frozen overlay, not evidence that raw
  capacity scaling is solved.

The next experiment should test whether the trained exact packet can be mixed
into recurrent query/router state with a small fixed scale, without replacing
the learned accumulator. That will distinguish an output-codec gain from a
genuine improvement to the engine's intermediate dataflow.

## Artifacts

- `train_integer_output_overlay.py`
- configs `...integeroverlay_codec_calibration.yaml`,
  `..._w025.yaml`, and `..._w010.yaml`, `..._w005.yaml`
- taskwise reports:
  `results/runs/v0_249_codec_calibrated_overlay_taskwise_seed17.json`,
  `...seed18.json`,
  `results/runs/v0_250_codec_calibrated_w025_overlay_taskwise_seed17.json`,
  `...seed18.json`,
  `results/runs/v0_251_codec_calibrated_w010_overlay_taskwise_seed17.json`,
  `...seed18.json`,
  `results/runs/v0_252_codec_calibrated_w005_overlay_taskwise_seed17.json`,
  `...seed18.json`
