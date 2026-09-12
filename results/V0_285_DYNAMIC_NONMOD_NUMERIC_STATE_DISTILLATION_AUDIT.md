# V0.285 — exact-packet logit distillation into the prior-free state

**Date:** 2026-09-12  
**Branch:** `exp/track-native-engine`  
**Status:** `REJECTED FOR ADOPTION; PRIOR-FREE STATE BOTTLENECK REMAINS`

## Question

V0.260/V0.261 reach the current quality ceiling with an exact integer packet
and a small frozen output overlay. This experiment asks whether the useful
numeric information can be transferred into the ordinary learned recurrent
path without retaining the exact packet at inference.

The frozen teacher is the corresponding V0.260/V0.261 peak checkpoint. The
student starts from the matched V0.264 prior-free checkpoint and has no
`algebraic_integer_output_decoder`. During training only, the student matches
the teacher's four digit distributions at every executed prefix stage using
temperature-2 KL, while retaining the normal terminal and stage hard-label
losses. Evaluation uses only the student path.

## Protocol

- student: V0.264 prior-free 300M virtual-bank configuration;
- teacher: corresponding V0.260/V0.261 exact-integer overlay checkpoint;
- same seed-specific student initialization for control and treatment;
- train values `0..95`, train depths `1..2`;
- held-out values `0..95`, held-out depths `3..4`;
- 2,000 continuation steps, batch `128`, CUDA;
- 1,024 identical evaluation examples per held-out depth;
- control: ordinary continuation, distillation weight `0`;
- treatment: prefix digit-logit distillation, weight `1`, temperature `2`;
- teacher is frozen and never used during evaluation.

## Results

| Arm | Seed | Held-out accuracy | Depth 3 | Depth 4 | CE |
|---|---:|---:|---:|---:|---:|
| Control | 17 | 9.277% | 12.207% | 6.348% | 12.4387 |
| Distillation | 17 | 8.740% | 11.230% | 6.250% | 14.7687 |
| Control | 18 | 10.010% | 13.672% | 6.348% | 11.8852 |
| Distillation | 18 | 11.133% | 15.332% | 6.934% | 14.1133 |
| **Control mean** | — | **9.644%** | **12.939%** | **6.348%** | **12.1620** |
| **Distillation mean** | — | **9.937%** | **13.281%** | **6.592%** | **14.4410** |

Treatment minus control is `+0.293 pp` accuracy, `+0.342 pp` depth-3, and
`+0.244 pp` depth-4, while CE regresses by `+2.2790`. The hard-accuracy change
is far below the project's `+2 pp` adoption gate and is not accompanied by a
calibration improvement. Seed17 regresses in hard accuracy while seed18 gains,
so the small mean movement is seed-unstable.

Both students report `student_has_integer_decoder: false`; the teacher is only
present in the training loop. The route audit remains the same sparse
active-8 path and no circuit-bank or router capacity was added.

## Interpretation

Directly matching the teacher's final digit distributions is not enough to
make the learned recurrent state carry the exact numeric contract. The teacher
signal is available, but it reaches the student through a weak terminal output
interface; it does not automatically become a reusable state transition.

This rejects this particular logit-distillation recipe, not all forms of
teacher transfer. A future transfer test would need to distill an explicit
intermediate state/transition representation or use a staged teacher-forcing
schedule, with a no-teacher evaluation gate. More model capacity is not
justified by this result.

## Decision

**Do not adopt V0.285 and do not scale it to 700M/1B.** Keep the script and
checkpoints as an opt-in diagnostic. The current peak exact overlay remains a
quality/reference ceiling, while the prior-free P-003/P-004 path remains open.
The next Native experiment should target an explicit learned numeric
transition/state contract rather than another output-logit loss or router
change.

## Artifacts

- `benchmark_numeric_state_distillation.py`
- `results/runs/v0_285_numeric_distill_control_seed17_2000.json`
- `results/runs/v0_285_numeric_distill_treatment_seed17_2000.json`
- `results/runs/v0_285_numeric_distill_control_seed18_2000.json`
- `results/runs/v0_285_numeric_distill_treatment_seed18_2000.json`
- corresponding checkpoints under `results/checkpoints/`
