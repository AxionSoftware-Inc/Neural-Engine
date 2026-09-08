# V0.200 DynamicRegister State-Trace Audit

## Question

At which recurrent step does the non-modular `0--7`, depth-4 failure begin?
Specifically, does the state still carry the intermediate accumulator value, or
does the output head merely learn a direct shortcut from the current input?

## Protocol

The frozen V0.198 flat-output checkpoints (300M-class DynamicRegister,
3,000-step training, seeds 17/18) were run with an inference-only trace of
`pre_accumulator`, pre-circuit `query`, `post_accumulator`, and final
`step_state`. A scalar linear probe was fitted on an all-depth calibration
stream and evaluated on held-out depths 3--4. The probe target is the true
intermediate accumulator before the `4096` class offset. Direct intermediate
head accuracy is reported as a reference. No model weights were changed.

## Results

The direct intermediate head and post-state linear probe show the same
depth-dependent degradation:

| stage | direct stage accuracy, mean | post-state correlation, seed17 | post-state correlation, seed18 |
|---:|---:|---:|---:|
| 0 | 100.00% | 1.000 | 1.000 |
| 1 | 100.00% | 0.905 | 0.888 |
| 2 | 83.89% | 0.503 | 0.533 |
| 3 | 66.70% | 0.207 | -0.012 |

The pre-state/query probes follow the same pattern: query correlation is
`0.708/0.755` at stage 1, `0.404/0.368` at stage 2, and `0.164/0.102` at
stage 3 for seeds 17/18. Post-state held-out R2 is positive at stages 0--2
for seed17 (`0.517/0.816/0.243`) but negative at stage 3; seed18 is already
negative at stages 2--3. The R2 values are sensitive to the wider stage-3
target distribution, so correlation and direct accuracy are the primary
summary.

## Interpretation

The first operation is represented and decoded correctly. After one more
operation, the recurrent state still contains a usable but imperfect scalar
signal. By the third and fourth operations, that signal is no longer stable
across seeds, while the model still fits depths 1--2 perfectly. This localizes
the current quality ceiling to repeated state transition/composition, not
only to the 32,768-class output head and not to an obvious dead-router event.

This is a diagnostic, not proof that all state information is absent: a
nonlinear probe or a different representation could recover more. It does
show that simply increasing the circuit bank or changing candidate retrieval
is premature for this task.

## Decision and next test

**State interface remains the primary active problem.** Keep V0.199's
factorized output as a sparse accounting option, but do not treat it as a
quality fix. The next opt-in experiment adds a small operation-conditioned
bilinear scalar lane, using learned coefficients over
`(old_value, operand, old_value * operand, bias)`, and projects it into the
existing query/state. This is not an arithmetic oracle; the coefficients are
learned. It will be compared against the factorized-output control with the
same 0--7 depth-4 protocol. Adoption requires a repeatable held-out gain,
not merely a stronger scalar probe.

## Artifacts

- `diagnose_dynamic_state.py`
- `neural_engine/dynamic_register.py` (`collect_state_stats` trace)
- `results/runs/nonmod_depth4_values0_7_state_probe.json`
- `results/checkpoints/nonmod_depth4_values0_7_typed_write_seed17_3000.pt`
- `results/checkpoints/nonmod_depth4_values0_7_typed_write_seed18_3000.pt`
