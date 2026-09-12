# V0.284 — V0.260/V0.261 peak overlay recheck

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Purpose

V0.255 was rechecked in V0.283 at `98.315%`. This audit repeats the same
current evaluator check for the longer-trained V0.260/V0.261 rank-128 exact
integer overlay checkpoints, which were previously reported as the peak
quality reference.

## Protocol

- checkpoints: `v0_260_rank128_fullrange_codec_continued_seed17_15000.pt` and
  `v0_261_rank128_fullrange_codec_continued_seed18_15000.pt`;
- held-out depths 3–4;
- values `0..95`;
- 1,024 identical examples per depth;
- evaluator seed `5102`;
- compact four-digit evaluation;
- no weights changed during recheck.

## Results

| Seed | Accuracy | CE | Depth 3 | Depth 4 |
|---:|---:|---:|---:|---:|
| 17 | 99.756% | 0.0172 | 99.902% | 99.609% |
| 18 | 99.902% | 0.0152 | 99.902% | 99.902% |
| **Mean** | **99.829%** | **0.0162** | **99.902%** | **99.756%** |

This is higher than the V0.255 5k overlay recheck (`98.315%`) and confirms
that the extra overlay training budget closes nearly all remaining in-range
errors.

## Interpretation and limitation

V0.260/V0.261 are the current quality ceiling, but they are not a proof of a
fully learned sparse Neural Engine. The exact integer packet supplies the
known add/subtract/multiply transition; the trainable overlay primarily learns
the terminal output codec. Earlier circuit ablations also found that removing
the learned circuit residual did not materially change this quality.

Therefore this result is a regression reference and a practical numeric
candidate, not evidence that increasing prior-free model capacity to 700M or
1B will solve the task.

## Decision

**RETAINED AS PEAK QUALITY REFERENCE.** The next research task is to compress
or distill the numeric representation into the learned recurrent state while
measuring the prior-free path separately. No new bank-scale claim follows.

## Raw runs

- `results/runs/v0_284_v0260_recheck_seed17_large.json`
- `results/runs/v0_284_v0260_recheck_seed18_large.json`

Checkpoints:

- `results/checkpoints/v0_260_rank128_fullrange_codec_continued_seed17_15000.pt`
- `results/checkpoints/v0_261_rank128_fullrange_codec_continued_seed18_15000.pt`
