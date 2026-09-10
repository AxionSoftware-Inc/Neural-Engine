# V0.210 Shared Low-Rank Value Codec

## Question

V0.209 established that the `0--31` quality path prefers a fine output base:
base 512 beat base 1024, but its full high-digit classifier consumed
`27.35M` estimated active parameters. Can a shared low-rank state projection
retain the fine-codec quality while removing redundant output-head weights?

## Architecture

The recurrent state, algebraic `x,x^2` packet, Fourier bridge, router, circuit
bank, data generator, and training budget are unchanged from the V0.209
base-512 arm. Only `FactorizedDigitOutput` changes: both digit classifiers
share a learned `384 -> 128` projection, then use separate `128 -> 65,536`
and `128 -> 512` classifiers. The full Cartesian output is still represented
exactly at inference by combining the two digit logits; compact factorized
training remains enabled.

This is a shared low-rank value codec, not a change to the circuit routing or
an increase in active paths.

## Protocol

The 300M-class factorized DynamicRegister uses ordinary non-modular arithmetic,
operand range `0--31`, training depths `1--2`, held-out depths `3--4`, target
offset `1,048,576`, batch size `128`, two seeds (`17`, `18`), and `3,000` CUDA
steps. The comparator arms use the same protocol: base-1024 full output,
base-512 full output, and base-512 rank-128 shared codec.

## Results

| variant | seed | held-out | depth 3 | depth 4 | train accuracy | total params | active estimate | train sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base 1024 full | 17 | 73.83% | 76.17% | 71.48% | 99.22% | 20.23M | 14.93M | 474.8 |
| base 1024 full | 18 | 76.37% | 84.38% | 68.36% | 99.22% | 20.23M | 14.93M | 488.3 |
| **base 1024 mean** | — | **75.10%** | **80.27%** | **69.92%** | **99.22%** | — | — | — |
| base 512 full | 17 | 78.32% | 80.47% | 76.17% | 99.02% | 32.65M | 27.35M | 577.0 |
| base 512 full | 18 | 79.30% | 87.11% | 71.48% | 99.61% | 32.65M | 27.35M | 566.7 |
| **base 512 full mean** | — | **78.81%** | **83.79%** | **73.83%** | **99.32%** | — | — | — |
| base 512 rank 128 | 17 | 86.91% | 87.89% | 85.94% | 98.83% | 15.79M | 10.49M | 501.2 |
| base 512 rank 128 | 18 | 83.20% | 90.23% | 76.17% | 99.80% | 15.79M | 10.49M | 503.4 |
| **rank-128 mean** | — | **85.06%** | **89.06%** | **81.05%** | **99.32%** | — | — | — |
| **rank 128 − base 1024** | — | **+9.96 pp** | **+8.79 pp** | **+11.14 pp** | **+0.10 pp** | **−4.44M** | **−4.44M** | **+21.8** |
| **rank 128 − base 512 full** | — | **+6.25 pp** | **+5.27 pp** | **+7.23 pp** | **0.00 pp** | **−16.86M** | **−16.86M** | **−69.5** |

The rank-128 codec improves held-out quality on both seeds. It cuts total and
estimated active parameters by `21.9%` and `29.8%` relative to base 1024,
while adding only about `4.5%` mean training time. Relative to the full
base-512 quality arm, it removes `51.8%` of total and `61.6%` of estimated
active parameters and is about `12%` faster.

## Interpretation

This is the strongest Native Engine signal so far. The previous base sweep
showed that output resolution matters; this experiment shows that the
resolution can be retained without paying for an independent full-width
state-to-high-digit matrix. The gain is not from more circuits: router and
factor bank settings are unchanged, and the rank-128 arm has fewer learned
output parameters.

The result still needs one robustness gate before changing any default:
evaluate shifted target offsets and a wider operand range, and run a small
rank sweep (`64`, `128`, `256`) on the same fixed protocol. The current two
seeds are sufficient to retain the architecture as the leading opt-in, but
not to claim a universal arithmetic codec.

## Decision

**RETAIN AS THE LEADING OPT-IN NATIVE ENGINE ARCHITECTURE.** Do not scale to
700M/1B yet and do not replace the existing default. The next work is
robustness and rank/codec ablation, followed by a checkpointed 0--31 quality
reference if the result remains stable.

## Artifacts

- `neural_engine/dynamic_register.py` (`output_factor_rank` and shared codec)
- `train_dynamic_composition.py`
- `tests/test_dynamic_register.py`
- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_31_factorized_algebraic_fourier_base512_rank128.yaml`
- `results/runs/nonmod_depth4_values0_31_factorized_algebraic_fourier_base512_rank128_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_31_factorized_algebraic_fourier_base512_rank128_seed18_3000.json`

