# V0.207 Algebraic State Range Stress: Operands 0--15

## Question

Does the positive `x,x^2` algebraic state primitive from V0.206 survive a
larger operand domain, or is its gain limited to the `0--7` screen?

## Protocol

The V0.206 300M-class factorized DynamicRegister configuration is kept
unchanged except for the operand/output range. Operands are expanded from
`0--7` to `0--15`; arithmetic is ordinary non-modular add/subtract/multiply,
train depths are `1--2`, held-out depths are `3--4`, and the target offset is
`65,536`. The exact output range is represented by a factorized
`524,288`-class head with base `128`. The algebraic packet uses normalized
`x,x^2` coordinates with scale `524,288`. Each seed is trained for `3,000`
CUDA steps with batch size `512`.

During this screen the training path was corrected to avoid materializing the
full factorized Cartesian logits; training uses only the compact digit logits,
while evaluation still reconstructs exact class logits for argmax and CE.
This changes compute, not the quality metric or model semantics.

## Results

| seed | train accuracy | held-out accuracy | depth 3 | depth 4 | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.00% | 65.63% | 76.17% | 55.08% | 8.83M | 3.53M |
| 18 | 99.90% | 63.28% | 75.20% | 51.37% | 8.83M | 3.53M |
| **mean** | **99.95%** | **64.45%** | **75.68%** | **53.22%** | — | — |

The same algebraic primitive on operands `0--7` reached `78.52%` mean held
out and `70.70%` mean depth-4 in V0.206. Expanding the operand domain reduces
held-out accuracy by `14.06 pp` and depth-4 accuracy by `17.48 pp`, even
though the model fits the seen depths almost perfectly. Evaluation factor-row
usage remains broad, so this is not a dead-router explanation.

## Interpretation

V0.206's large gain is real on the bounded `0--7` task, but it is not yet a
scale-free reusable state solution. The wider screen increases the output
space to `524,288` classes and the active estimate to `3.53M`; more
importantly, the learned readout still fails to transfer the exact algebraic
state through deeper compositions. The compact packet preserves the numeric
value, but the current learned circuit/readout interface does not exploit it
reliably over the larger dynamic range.

This closes the immediate path of simply increasing the normalization scale
or operand range around the same two-coordinate packet. It does not invalidate
the packet as a diagnostic or forbid a genuinely range-aware readout.

## Decision

**REJECTED AS A UNIVERSAL QUALITY/SCALE SOLUTION; RETAINED AS A BOUNDED
DIAGNOSTIC.** Do not scale the current polynomial2 packet to 500M/700M/1B and
do not claim the V0.206 gain as general model quality. The next useful test is
a range-aware value codec/readout that preserves algebraic state identity
without a giant flat or Cartesian output burden, with an exact matched
control. If that fails, return to the Qwen/FFN transplant lane rather than
adding more generic circuit capacity.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_15_factorized_algebraic_state.yaml`
- `neural_engine/dynamic_register.py` (`algebraic_state_mode: polynomial2`)
- `train_dynamic_composition.py` (compact factorized training path)
- `results/runs/nonmod_depth4_values0_15_factorized_algebraic_state_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_15_factorized_algebraic_state_seed18_3000.json`
