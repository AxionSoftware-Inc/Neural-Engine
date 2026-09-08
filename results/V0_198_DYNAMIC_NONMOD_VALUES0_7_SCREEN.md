# V0.198 Non-Modular Values 0--7 Depth-4 Screen

## Question

Can the positive 0--3 non-modular depth-generalization signal survive a
larger operand domain without changing the typed-write state interface?

## Protocol

The 300M-class DynamicRegister configuration keeps train depths 1--2 and
held-out depths 3--4, but expands every operand from 0--3 to 0--7. Arithmetic
is ordinary integer add/subtract/multiply (`modulus: null`). The maximum
possible target is kept in range with offset 4096 and a 32,768-class learned
output head. Batch size is 512 and the screen runs 3,000 steps on CUDA.

The first attempt used offset 64 and failed before evaluation because negative
depth-4 products produced invalid class labels. It is invalid and excluded.
The corrected run includes target-range validation.

## Results

| seed | train depth 1--2 | held-out depth 3--4 | depth 3 | depth 4 | total params | active estimate |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.00% | 72.27% | 81.25% | 63.28% | 19.82M | 14.52M |
| 18 | 100.00% | 77.34% | 85.16% | 69.53% | 19.82M | 14.52M |
| **mean** | **100.00%** | **74.80%** | **83.20%** | **66.41%** | — | — |

The same 300M typed-write state path on values 0--3 reached `84.70%` mean at
9,000 steps across three seeds. The 0--7 3k mean is therefore `9.90 pp`
lower, while the output head grows from 512 to 32,768 classes and dominates
the estimated active path. Both runs fit seen depths perfectly and use the
factor rows broadly; this is not a dead-router failure.

## Decision

**REJECTED AS A SPARSE QUALITY/SCALE CONFIGURATION.** The larger numeric domain
reveals two coupled limits: the recurrent interface does not transfer the
same quality to wider values, and a flat classifier makes the active path
mostly dense output parameters. Increasing the circuit bank would not isolate
either problem and is not justified.

The next test is a factorized digit output with the same 32,768 label space but
small high/low digit heads. It will be evaluated with exact reconstructed-class
accuracy and separate output compute accounting. If it improves quality while
returning the active estimate near 2M, the output interface—not raw circuit
capacity—was a major part of the observed ceiling.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_7.yaml`
- `results/runs/nonmod_depth4_values0_7_typed_write_seed17_3000.json`
- `results/runs/nonmod_depth4_values0_7_typed_write_seed18_3000.json`
- `train_dynamic_composition.py` target-range validation
