# V0.212 Range-63 Depth-3 Gate

## Question

V0.211 passed the offset robustness check, but extending operands to `0--63`
with the old depth-4/two-digit class space exposed a protocol limit: random
depth-4 programs can produce targets above `225M`, outside the `67M` classifier
range. Before changing the model, this audit measures the same rank-128 codec
on a valid `0--63` depth-3 gate.

## Protocol

The shared rank-128 base-512 codec is unchanged. The valid range-63 gate uses
`max_ops=3`, train depths `1--2`, held-out depth `3`, target offset
`1,048,576`, `num_classes=33,554,432`, batch size `128`, and `3,000` CUDA
steps. Two seeds (`17`, `18`) are evaluated. This avoids materializing an
invalid classifier for the larger depth-4 target range.

## Results

| seed | held-out depth 3 | train accuracy | total params | active estimate | train sec |
|---:|---:|---:|---:|---:|---:|
| 17 | 81.64% | 98.24% | 15.79M | 10.49M | 479.2 |
| 18 | 89.84% | 98.05% | 15.79M | 10.49M | 478.1 |
| **mean** | **85.74%** | **98.14%** | — | — | — |

The range-63 mean is `3.32` percentage points below the matched `0--31`
rank-128 depth-3 mean (`89.06%`), but remains a strong non-collapse signal.
The router/factor-bank settings and active circuit budget are unchanged.

## Interpretation

The algebraic/Fourier state and shared codec continue to work when the
operand range doubles. The remaining depth-4 blocker is representation of the
larger integer target space, not evidence that the circuit bank must be
scaled. A two-digit base-512 output has one large quotient head when the
classifier covers a billion-class depth-4 space; that is the wrong sparse
codec. The next implementation is a three-digit factorized output with three
small heads and shared low-rank state projection.

## Decision

**RANGE-63 DEPTH-3 GATE PASSED AS A NON-COLLAPSE CONTROL.** Do not treat the
failed depth-4 class-range run as a model-quality result. Add the three-digit
codec and re-run `0--63` depth 4 with the same seeds and active-path budget.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth3_typed_write_adapter_values0_63_factorized_algebraic_fourier_base512_rank128.yaml`
- `results/runs/nonmod_depth3_values0_63_factorized_algebraic_fourier_base512_rank128_seed17_3000.json`
- `results/runs/nonmod_depth3_values0_63_factorized_algebraic_fourier_base512_rank128_seed18_3000.json`
- `results/runs/nonmod_depth4_values0_63_factorized_algebraic_fourier_base512_rank128_seed17_3000.json` (invalid class-space diagnostic; not adopted)
