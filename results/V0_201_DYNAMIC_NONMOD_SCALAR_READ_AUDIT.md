# V0.201 Persistent Scalar Read Audit

## Question

V0.200 showed that the intermediate-value signal decays in the recurrent
state. Does the learned operation-conditioned scalar lane become useful if it
is also injected into the next operation's `read_accumulator`, rather than
only into the query and output readout?

## Protocol

The same 300M-class, attention-free DynamicRegister configuration is used for
ordinary integer arithmetic on values `0--7`, training depths `1--2`, held-out
depths `3--4`, target offset `4096`, factorized output digits, batch size `512`,
and `3,000` CUDA steps. The treatment adds a learned scalar state with
operation-specific coefficients over `(old, operand, old*operand, bias)` and
injects its projected value into the next operation's read accumulator with
scale `1.0`. No circuit-bank or router capacity is changed.

The preceding scalar-only variant is included to separate “scalar lane exists”
from “scalar lane is injected into the next read”.

## Results

| variant | seed | held-out depth 3--4 | depth 3 | depth 4 |
|---|---:|---:|---:|---:|
| factorized output control | 17 | 71.09% | 82.81% | 59.38% |
| factorized output control | 18 | 72.85% | 85.16% | 60.55% |
| **control mean** | — | **71.97%** | **83.98%** | **59.96%** |
| scalar lane, query/output only | 17 | 73.63% | 82.81% | 64.45% |
| scalar lane, query/output only | 18 | 71.09% | 83.59% | 58.59% |
| **scalar-only mean** | — | **72.36%** | **83.20%** | **61.52%** |
| scalar lane + persistent read | 17 | 72.27% | 82.03% | 62.50% |
| scalar lane + persistent read | 18 | 71.09% | 82.42% | 59.77% |
| **persistent-read mean** | — | **71.68%** | **82.23%** | **61.13%** |

The persistent-read treatment is `-0.29 pp` below its factorized control,
`-1.76 pp` on depth 3, and `+1.17 pp` on depth 4. The scalar-only treatment
has a seed17 gain but only `+0.39 pp` mean and a seed18 regression.

## Decision

**REJECTED FOR ADOPTION.** Adding the learned scalar signal to the next
operation's read path does not produce a repeatable gain. The failure is not
fixed by another scalar injection point; the dense recurrent writer still
does not preserve a stable compositional value contract across depth.

The scalar lane remains opt-in research code because it is a useful diagnostic
and costs very little, but it is not a reason to scale the bank or move to
700M/1B. The factorized output branch also remains a sparsity accounting
option, not a quality default.

## Next action

Stop repeating scalar read/scale variants. The next architectural test should
make the algebraic register persistent and authoritative for state transition
(a separate value packet/transition path with a controlled handoff to the
dense circuit state), or else close this task as an interface ceiling. It must
be compared against the same factorized control and must report whether the
intermediate value survives to stage 3/4.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_7_factorized_structured_scalar.yaml`
- `neural_engine/dynamic_register.py` (`structured_scalar_read_scale`)
- `results/runs/nonmod_depth4_values0_7_factorized_structured_scalar_read_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_7_factorized_structured_scalar_read_seed18_3000.json`
