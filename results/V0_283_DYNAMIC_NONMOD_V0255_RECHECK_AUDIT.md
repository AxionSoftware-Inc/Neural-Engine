# V0.283 — V0.255 exact-integer overlay recheck

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Purpose

The recent prior-free and curriculum screens clarify that the learned sparse
circuit path is still weak on arithmetic composition. V0.255 was previously
the strongest quality checkpoint because it used a frozen exact-integer
algebraic overlay with a calibrated output codec. This audit reruns that
checkpoint through the current 1,024-example paired evaluator so the project
has a directly comparable quality reference.

## Protocol

- checkpoints: `v0_255_fullrange_codec_seed17_5000.pt` and seed 18;
- held-out depths 3–4;
- values `0..95`;
- 1,024 identical examples per depth;
- evaluator seed `5102`;
- compact four-digit output evaluation;
- no weights changed during recheck.

## Results

| Seed | Accuracy | CE | Depth 3 | Depth 4 | Factor rows |
|---:|---:|---:|---:|---:|---:|
| 17 | 98.291% | 0.0880 | 99.121% | 97.461% | 111 |
| 18 | 98.340% | 0.0869 | 99.121% | 97.559% | 79 |
| **Mean** | **98.315%** | **0.0875** | **99.121%** | **97.510%** | — |

For context, the current prior-free V0.273/V0.276 control mean is `7.031%`
under its paired evaluation, and the best curriculum result in the same
wide-range family is `20.422%` when values `0..95` are included in training.
The gap is real, but it is not a fair learned-vs-learned comparison: V0.255's
exact integer packet supplies the known add/subtract/multiply transition and
the trained overlay mainly learns the output codec.

## Decision

**RETAINED AS THE CURRENT QUALITY REFERENCE, NOT AS PROOF OF A FULLY LEARNED
SPARSE CIRCUIT.** Use V0.255 as a regression ceiling and deployment-oriented
numeric candidate, while reporting prior-free results separately. Do not infer
that 700M/1B learned capacity will reproduce `98%` without the exact packet.

The next engineering target is to reduce the exact overlay's learned budget
or distill only its reusable numeric representation into the Neural Engine's
sparse state path, with the prior-free control and unseen-value gate retained.

## Raw runs

- `results/runs/v0_283_v0255_recheck_seed17_large.json`
- `results/runs/v0_283_v0255_recheck_seed18_large.json`

Checkpoints:

- `results/checkpoints/v0_255_fullrange_codec_seed17_5000.pt`
- `results/checkpoints/v0_255_fullrange_codec_seed18_5000.pt`
