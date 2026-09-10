# V0.211 Shared Low-Rank Codec Offset Robustness

## Question

V0.210's shared rank-128 base-512 codec produced the strongest quality and
active-budget combination so far. This audit checks whether the result depends
on the exact target offset used during training.

## Protocol

The model is unchanged from V0.210: 300M-class factorized DynamicRegister,
`polynomial2_fourier` state, Fourier/output base `512`, shared output rank
`128`, operand range `0--31`, train depths `1--2`, held-out depths `3--4`,
batch size `128`, and `3,000` CUDA steps. The only change is target offset:
`1,048,576` in the original arm and `2,097,152` in the shifted arm. Both use
seeds `17` and `18`.

## Results

| target offset | seed | held-out | depth 3 | depth 4 | train accuracy | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,048,576 | 17 | 86.91% | 87.89% | 85.94% | 98.83% | 15.79M | 10.49M |
| 1,048,576 | 18 | 83.20% | 90.23% | 76.17% | 99.80% | 15.79M | 10.49M |
| **original mean** | — | **85.06%** | **89.06%** | **81.05%** | **99.32%** | — | — |
| 2,097,152 | 17 | 85.35% | 85.55% | 85.16% | 98.83% | 15.79M | 10.49M |
| 2,097,152 | 18 | 84.38% | 89.06% | 79.69% | 100.00% | 15.79M | 10.49M |
| **shifted mean** | — | **84.86%** | **87.30%** | **82.42%** | **99.41%** | — | — |
| **shifted − original** | — | **−0.20 pp** | **−1.76 pp** | **+1.37 pp** | **+0.10 pp** | **0** | **0** |

The shifted offset preserves held-out quality within `0.20` percentage points
overall and improves the depth-4 mean by `1.37` points. Parameter counts and
active estimates are identical by construction.

## Interpretation

The shared codec is not tied to one absolute target offset. The remaining
variation is normal two-seed/task variation, not a collapse of the readout
interface. This strengthens the V0.210 result, but it does not establish
generalization to a wider operand range or arbitrary operations; the Fourier
periods and factorized output are still a structured codec hypothesis.

## Decision

**ROBUSTNESS GATE PASSED FOR TARGET-OFFSET SHIFT.** Retain the rank-128 shared
codec as the leading opt-in architecture. The next gate is operand range
`0--63` with the same model and budget; do not increase model scale first.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_31_factorized_algebraic_fourier_base512_rank128_offset2097152.yaml`
- `results/runs/nonmod_depth4_values0_31_factorized_algebraic_fourier_base512_rank128_offset2097152_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_31_factorized_algebraic_fourier_base512_rank128_offset2097152_seed18_3000.json`

