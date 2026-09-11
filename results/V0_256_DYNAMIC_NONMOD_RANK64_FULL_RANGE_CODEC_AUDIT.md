# V0.256 — rank64 full-range integer codec screen

**Date:** 2026-09-11  
**Status:** **REJECTED FOR QUALITY; ONE-SEED EARLY STOP**

## Question

V0.255 is the leading full-range all-operation integer codec, but its
factorized output head uses rank128 and `337,474` trainable overlay
parameters. This screen asks whether rank64 can preserve the quality while
cutting the overlay to `207,362` trainable parameters.

The frozen body, router, circuit bank, codec range, and training protocol are
unchanged. The rank128 integer output head is not shape-compatible with the
rank64 head, so the head is freshly initialized; compatible decoder and digit
embedding weights are reused, and the mismatched head keys are explicitly
reported by the trainer.

## Protocol

- initialization: V0.252 overlay checkpoint, compatible codec weights reused
- seed screened: `17`
- overlay training: `5,000` steps, batch `128`
- train values: `0..95`; held-out depths: `3..4`
- matched evaluation: `256` examples per depth, values `0..95`
- unseen evaluation: fixed operand `96`
- full model: `7,708,105` parameters
- trainable rank64 overlay: `207,362` parameters

## Results

| Range / metric | All ops | Depth 3 | Depth 4 | Add | Subtract | Multiply | Mul d3 | Mul d4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Matched `0..95` | 90.4297% | 90.6250% | 90.2344% | 100.0000% | 81.0547% | 92.7734% | 92.9688% | 92.5781% |
| Unseen fixed `96` | 91.7969% | 92.5781% | 91.0156% | 100.0000% | 50.0000% | 100.0000% | 100.0000% | 100.0000% |

The matched result fails the preservation requirement because subtract falls
from V0.255's `100%` to `81.05%`; unseen subtract is `50%`. The aggregate is
also far below V0.255's two-seed `98.05%`.

## Decision

**Rejected for quality.** Rank64 retains a strong multiply signal but cannot
preserve the universal all-operation behavior. No second seed was run after
the first seed violated the hard add/subtract quality gate. Rank128 remains
the leading opt-in overlay budget for this line; rank32 is not worth a full
screen under the same gate.

This is a capacity result for the terminal codec head only. It does not imply
that the 300M body or sparse router needs rank128; it says that this particular
low-rank output readout loses information when compressed directly from rank128
to rank64.

## Artifact

- config: `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank64_interaction16_hardcontext_integeroverlay_codec_calibration_w005_all_fullrange.yaml`
- checkpoint: `results/checkpoints/v0_256_rank64_fullrange_codec_seed17_5000.pt`
- taskwise report: `results/runs/v0_256_rank64_fullrange_codec_taskwise_seed17.json`
