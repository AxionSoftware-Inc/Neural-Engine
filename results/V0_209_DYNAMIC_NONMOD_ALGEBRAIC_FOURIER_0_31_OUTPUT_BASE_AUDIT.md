# V0.209 0--31 Algebraic/Fourier Output-Base Audit

## Question

V0.208 showed a strong gain on operands `0--15` after adding a range-aware
Fourier bridge to the fixed `x,x^2` state. The next gate was `0--31`. The
first `0--31` control used the same `output_digit_base=1024` interface as the
range-aware codec. This audit tests whether reducing the number of output
classes per digit with `output_digit_base=4096` preserves quality while
cutting the output-head cost.

## Protocol

Both arms use the same 300M-class factorized DynamicRegister, ordinary
non-modular arithmetic, target offset `1,048,576`, operand range `0--31`,
training depths `1--2`, held-out depths `3--4`, two seeds (`17`, `18`), batch
size `128`, and `3,000` CUDA steps. The algebraic state is
`polynomial2_fourier`; the Fourier base is matched to the output digit base.
Compact factorized training/evaluation avoids materializing the full
`33,554,432`-class Cartesian logit matrix.

## Results

| variant | seed | held-out | depth 3 | depth 4 | train accuracy | total params | active estimate | train sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| output base 1024 | 17 | 73.83% | 76.17% | 71.48% | 99.22% | 20.23M | 14.93M | 474.8 |
| output base 1024 | 18 | 76.37% | 84.38% | 68.36% | 99.22% | 20.23M | 14.93M | 488.3 |
| **base 1024 mean** | — | **75.10%** | **80.27%** | **69.92%** | **99.22%** | — | — | — |
| output base 2048 | 17 | 67.77% | 71.09% | 64.45% | 98.63% | 14.32M | 9.02M | 435.5 |
| output base 2048 | 18 | 71.88% | 82.42% | 61.33% | 98.83% | 14.32M | 9.02M | 450.9 |
| **base 2048 mean** | — | **69.82%** | **76.76%** | **62.89%** | **98.73%** | — | — | — |
| output base 512 | 17 | 78.32% | 80.47% | 76.17% | 99.02% | 32.65M | 27.35M | 577.0 |
| output base 512 | 18 | 79.30% | 87.11% | 71.48% | 99.61% | 32.65M | 27.35M | 566.7 |
| **base 512 mean** | — | **78.81%** | **83.79%** | **73.83%** | **99.32%** | — | — | — |
| output base 4096 | 17 | 60.35% | 67.19% | 53.52% | 97.46% | 11.95M | 6.65M | 444.5 |
| output base 4096 | 18 | 64.26% | 75.39% | 53.13% | 97.66% | 11.95M | 6.65M | 424.3 |
| **base 4096 mean** | — | **62.30%** | **71.29%** | **53.32%** | **97.56%** | — | — | — |
| **2048 − 1024** | — | **−5.28 pp** | **−3.51 pp** | **−7.03 pp** | — | **−5.91M** | **−5.91M** | **−38.4** |
| **512 − 1024** | — | **+3.71 pp** | **+3.52 pp** | **+3.91 pp** | — | **+12.42M** | **+12.42M** | **+90.3** |
| **4096 − 1024** | — | **−12.80 pp** | **−8.98 pp** | **−16.60 pp** | — | **−8.28M** | **−8.28M** | **−37.2** |

The base-2048 arm reduces total and estimated active parameters by about
`29%`, and training time by about `8%`, but loses `5.28` percentage points on
held-out accuracy and `7.03` points at depth 4. Base 4096 reduces the budget
by about `41%` and loses `12.80`/`16.60` points. Both seeds show the same
quality direction. Base 512 is the first lower-base setting to beat the
base-1024 reference: it gains `3.71`/`3.91` points, but costs `61%` more total
parameters, `83%` more estimated active parameters, and about `19%` more
training time. The quality ordering is consistent across two seeds, so this
is not a seed-only fluctuation.

## Interpretation

The result supports base 512 as the current quality reference for the `0--31`
gate, but not as the final efficiency solution. Base 1024 is the current
compute/quality compromise; base 2048 and 4096 are inferior on quality, while
base 512 exposes that more output resolution can recover depth transfer at a
substantial cost. The remaining engineering target is therefore a
low-rank/shared value codec, not a larger circuit bank.
The failure is not evidence that the algebraic state is useless: the
base-1024 `0--31` arm remains above the earlier polynomial2 stress result,
and the base-128 Fourier bridge already produced a large `0--15` gain. It
does show that output factorization is part of the value-to-readout contract;
coarser digits cannot be treated as a free compression of the same learned
codec.

The route audit remains broad in both arms, so this comparison does not point
to router starvation as the primary cause. The base-4096 arm also fits the
seen depths well (`97.56%` mean) while losing depth transfer, which is
consistent with a readout/composition interface mismatch rather than simple
under-training.

## Decision

**RETAIN base 512 as an opt-in quality reference; do not make it default.**
Base 2048 and base 4096 are rejected for quality adoption. Base 1024 remains
the efficiency reference, while base 512 is the quality reference for further
work. Do not increase model scale. The next experiment is a shared/low-rank
factorized value codec that targets base-512 quality with a base-1024-sized
active output budget.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_31_factorized_algebraic_fourier.yaml`
- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_31_factorized_algebraic_fourier_base512.yaml`
- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_31_factorized_algebraic_fourier_base2048.yaml`
- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_31_factorized_algebraic_fourier_base4096.yaml`
- `results/runs/nonmod_depth4_values0_31_factorized_algebraic_fourier_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_31_factorized_algebraic_fourier_seed18_3000.json`
- `results/runs/nonmod_depth4_values0_31_factorized_algebraic_fourier_base2048_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_31_factorized_algebraic_fourier_base2048_seed18_3000.json`
- `results/runs/nonmod_depth4_values0_31_factorized_algebraic_fourier_base512_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_31_factorized_algebraic_fourier_base512_seed18_3000.json`
- `results/runs/nonmod_depth4_values0_31_factorized_algebraic_fourier_base4096_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_31_factorized_algebraic_fourier_base4096_seed18_3000.json`
