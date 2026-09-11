# V0.269: learned numeric vector-state screen

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Purpose

V0.268 closed the one-dimensional scalar contract. V0.269 tests the existing
16-dimensional learned numeric scratch state as a vector-valued alternative.
It is initialized from normalized operands, updated by a learned operation-
conditioned transition, and projected into the recurrent query/output path.
No exact arithmetic packet or modular prior is enabled.

## Protocol

Config: `configs/ne_dynamic_300m_nonmod_small_values0_7_numeric_state16.yaml`

- same prior-free 300M-virtual body as V0.268;
- numeric state width 16, input scale 128;
- operands `0..7`, non-modular target offset `2**27`;
- train depths 1–2, held-out depths 3–4;
- 2,000 steps, batch size 128, seeds 17 and 18;
- matched control disables the numeric state;
- 256 examples per depth in the training reports;
- paired evaluation uses the same 1,024 held-out examples per depth.

The lane adds 8,736 stored parameters (`7,483,463 → 7,492,199`).

## Results

### Original held-out reports

| variant | seed | train d1–2 | held-out all | d3 | d4 |
|---|---:|---:|---:|---:|---:|
| control | 17 | 100.000% | 67.188% | 75.781% | 58.594% |
| control | 18 | 100.000% | 68.359% | 81.641% | 55.078% |
| numeric state 16 | 17 | 100.000% | 67.383% | 78.125% | 56.641% |
| numeric state 16 | 18 | 100.000% | 67.773% | 79.688% | 55.859% |

The nominal two-seed mean is `67.773%` for control and `67.578%` for the
numeric lane (`−0.195 pp`).

### Larger paired held-out evaluation

| variant | seed | all | d3 | d4 | eval loss |
|---|---:|---:|---:|---:|---:|
| control | 17 | 64.258% | 75.488% | 53.027% | 6.2236 |
| control | 18 | 68.164% | 79.102% | 57.227% | 5.2357 |
| numeric state 16 | 17 | 66.992% | 77.539% | 56.445% | 5.1086 |
| numeric state 16 | 18 | 68.994% | 79.004% | 58.984% | 4.8176 |

The two-seed means are `66.211%` for control and `67.993%` for the numeric
lane (`+1.782 pp`). This is a near-gate small-domain signal, but it is not
stable enough to adopt yet: the original evaluation is flat/slightly negative
and the effect is not reproduced on the wide-support task below.

Raw reports:

- `results/runs/v0_268_small07_control_seed17_2000.json`
- `results/runs/v0_268_small07_control_seed18_2000.json`
- `results/runs/v0_269_small07_numeric_state16_seed17_2000.json`
- `results/runs/v0_269_small07_numeric_state16_seed18_2000.json`
- `results/runs/v0_269_small07_paired_large_eval.json`

## Decision

**V0.269 is retained as an opt-in small-domain diagnostic, not adopted.**
The 16D vector lane avoids the severe scalar-contract regression and shows a
small depth-4 signal, but it does not yet meet the quality gate. The next
wide-support control is V0.270; no dimension sweep or 700M/1B scaling is
justified by this screen alone.
