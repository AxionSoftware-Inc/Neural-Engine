# V0.197 Non-Modular Depth-4 Typed-Write Third Seed

## Question

V0.196 validated a positive two-seed signal for typed-write DynamicRegister
generalization from training depths 1--2 to unseen non-modular depths 3--4.
This run adds seed19 without changing the model, data ranges, or schedule.

## Results

| seed | train depth 1--2 | held-out depth 3--4 | depth 3 | depth 4 |
|---:|---:|---:|---:|---:|
| 17 | 100.00% | 86.33% | 92.19% | 80.47% |
| 18 | 100.00% | 85.55% | 91.80% | 79.30% |
| 19 | 100.00% | 82.23% | 89.45% | 75.00% |
| **mean** | **100.00%** | **84.70%** | **91.15%** | **78.26%** |

Seed19 is `3.71 pp` below the two-seed mean, so variance is material. It is
still well above chance and all three seeds fit the seen depths perfectly.
There was no NaN or routing collapse; the 154 factor rows remained broadly
used on the held-out depth evaluation.

## Decision

**POSITIVE BUT NOT YET STABLE ENOUGH FOR CAPACITY SCALING.** The third seed
supports that the typed-write interface carries useful computation beyond the
training depth, but the depth-4 spread (`75.00%` to `80.47%`) is too large to
claim a strong scaling law or to justify 500M/700M/1B. The bottleneck remains
the repeated non-modular state transition, not lack of virtual bank rows.

The next stress test keeps the same 300M architecture and depth protocol but
expands operands from 0--3 to 0--7. This tests whether the signal is a real
state/composition interface or a small-value interpolation effect.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter.yaml`
- `results/runs/nonmod_depth4_typed_write_seed19_9000.json`
- `results/V0_196_DYNAMIC_NONMOD_DEPTH4_TYPED_WRITE_VALIDATION.md`
