# V0.277 — Value-curriculum generator-reset control

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Purpose

V0.274 showed a large gain from the schedule `0..7 → 0..31 → 0..95`.
Because the implementation recreates the training generator at each stage,
this control isolates a possible confound: perhaps generator resets alone,
rather than progressive value difficulty, caused the improvement.

## Control design

The model, optimizer, step budget, held-out-depth split, and evaluator are the
same as V0.274. The only schedule difference is that every stage uses the
full `0..95` range:

```yaml
value_curriculum:
  - until_step: 1000
    value_min: 0
    value_max: 95
  - until_step: 2500
    value_min: 0
    value_max: 95
  - until_step: 5000
    value_min: 0
    value_max: 95
```

This preserves the same stage boundaries and generator reinitialization while
removing the progressive-range treatment. Four seeds were trained for 5,000
steps and evaluated with 1,024 identical examples per held-out depth.

## Results

| Seed | Accuracy | CE | Depth 3 | Depth 4 |
|---:|---:|---:|---:|---:|
| 17 | 5.811% | 12.0072 | 8.008% | 3.613% |
| 18 | 6.982% | 11.8197 | 9.473% | 4.492% |
| 19 | 6.982% | 10.7370 | 8.691% | 5.273% |
| 20 | 6.152% | 11.3990 | 8.301% | 4.004% |
| **Mean** | **6.482%** | **11.4907** | **8.618%** | **4.346%** |

Relative to the V0.274 curriculum mean, the phase-reset control is
`−13.940 pp` in accuracy, `+1.8902` CE, `−17.944 pp` at depth 3, and
`−9.937 pp` at depth 4. It is also slightly below the ordinary full-range
control mean (`7.031%`).

## Decision

**GENERATOR RESET ALONE IS REJECTED AS THE EXPLANATION.** The positive V0.274
signal is attributable primarily to progressive value-range exposure, not to
restarting the sampler at steps 1,000 and 2,500. Keep the reset control in
future curriculum screens, but do not adopt it as a standalone technique.

This experiment does not prove that the curriculum is universally correct;
it only removes a concrete alternative explanation. V0.274 remains an opt-in
training protocol and the architecture/default path is unchanged.

Raw paired JSONs:

- `results/runs/v0_277_paired_value_reset_control_seed17_large.json`
- `results/runs/v0_277_paired_value_reset_control_seed18_large.json`
- `results/runs/v0_277_paired_value_reset_control_seed19_large.json`
- `results/runs/v0_277_paired_value_reset_control_seed20_large.json`
