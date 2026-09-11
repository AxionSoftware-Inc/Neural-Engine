# V0.270: wide-support numeric vector-state control

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Purpose

V0.269 showed a small `0..7` depth-transfer signal from a learned 16D numeric
state. V0.270 checks whether that signal survives the actual wide arithmetic
range that exposed the capacity/state problem: operands `0..95`, non-modular
targets, and depths 3–4 held out.

## Protocol

Configs:

- control: `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_prior_free.yaml`;
- treatment: `configs/ne_dynamic_300m_nonmod_train0_95_eval0_95_four_digit_base512_rank128_prior_free_numeric_state16.yaml`.

Both use the same 300M-virtual prior-free body, factorized four-digit output,
depths 1–2 for training, depths 3–4 held out, batch size 128, 2,000 steps,
and seeds 17/18. The treatment only adds the learned numeric state width 16;
there is no exact codec, algebraic state, or modular prior. A paired
evaluation then used 1,024 identical examples per held-out depth for all four
checkpoints.

## Results

### Larger paired held-out evaluation

| variant | seed | all | d3 | d4 | eval loss |
|---|---:|---:|---:|---:|---:|
| control | 17 | 1.855% | 2.246% | 1.465% | 10.8570 |
| control | 18 | 1.904% | 2.734% | 1.074% | 10.6638 |
| numeric state 16 | 17 | 1.807% | 2.344% | 1.270% | 10.3019 |
| numeric state 16 | 18 | 1.953% | 2.051% | 1.855% | 10.7966 |

The two-seed means are exactly `1.880%` for both control and treatment:
`0.000 pp` quality change. The treatment lowers loss in seed17 but does not
convert that into hard accuracy, and seed18 shows no consistent depth-wise
gain. The original 256-example reports also showed treatment below control
(`1.172%/1.172%` versus `1.367%/1.562%`).

Raw reports:

- `results/runs/v0_270_wide_control_seed17_2000.json`
- `results/runs/v0_270_wide_control_seed18_2000.json`
- `results/runs/v0_270_wide_numeric_state16_seed17_2000.json`
- `results/runs/v0_270_wide_numeric_state16_seed18_2000.json`
- `results/runs/v0_270_wide_paired_large_eval_all.json`

## Decision

**V0.270 rejects the plain numeric scratch state as a wide-range quality
solution.** A learned vector state that helps on `0..7` does not scale to
`0..95`; the wide-range bottleneck is a missing compositional/carry
representation, not merely insufficient state width. Keep numeric-state16 as
opt-in research code, do not widen it, and do not move to 700M/1B from this
result.

The next architecture must make carry/digit structure explicit while keeping
the sparse circuit path active, rather than adding another generic dense state
channel.
