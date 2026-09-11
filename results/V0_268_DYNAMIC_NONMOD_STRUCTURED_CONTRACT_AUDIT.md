# V0.268: supervised scalar value-contract screen

Date: 2026-09-12  
Branch: `exp/track-native-engine`

## Purpose

After V0.266/267 localized the failure to learned value/state composition, the
next low-cost test was a supervised contract at the recurrent boundary. This
variant uses the existing scalar lane: each operation predicts
`old + operand + old*operand + bias` with learned operation-specific
coefficients, feeds the resulting scalar back into the query/read path, and
receives a normalized intermediate-value loss. It is not an exact integer
prior, but it is deliberately a very strong scalar hypothesis and therefore a
useful falsification test before implementing a larger vector/carry format.

## Protocol

Config: `configs/ne_dynamic_300m_nonmod_small_values0_7_structured_contract.yaml`

- same 300M-virtual prior-free body and factorized four-digit output as V0.266;
- operands `0..7`, non-modular target offset `2**27`;
- train depths 1–2, held-out depths 3–4;
- 2,000 steps, batch size 128, seeds 17 and 18;
- treatment: `structured_scalar_state=true`, query/read scale `1.0`,
  normalized contract-loss weight `1.0`, target scale `64.0`;
- control: the matched prior-free model with the scalar lane disabled;
- 256 examples per depth in the training reports;
- a second evaluation-only pass used the same 1,024 held-out examples per
  depth for all four checkpoints.

The scalar lane adds only 780 stored parameters (`7,483,463 → 7,484,243`).
This is a representation/interface test, not a capacity test.

## Results

### Original 256-example held-out reports

| variant | seed | train d1–2 | held-out all | d3 | d4 |
|---|---:|---:|---:|---:|---:|
| control | 17 | 100.000% | 67.188% | 75.781% | 58.594% |
| control | 18 | 100.000% | 68.359% | 81.641% | 55.078% |
| supervised scalar contract | 17 | 98.633% | 55.273% | 65.234% | 45.313% |
| supervised scalar contract | 18 | 99.023% | 54.492% | 67.969% | 41.016% |

### Larger paired held-out evaluation

| variant | seed | held-out all | d3 | d4 | eval loss |
|---|---:|---:|---:|---:|---:|
| control | 17 | 67.822% | 78.320% | 57.324% | 5.6795 |
| control | 18 | 70.020% | 80.762% | 59.277% | 4.8469 |
| supervised scalar contract | 17 | 56.396% | 68.457% | 44.336% | 7.8016 |
| supervised scalar contract | 18 | 56.592% | 68.848% | 44.336% | 8.2489 |

The two-seed mean is `68.921%` for control and `56.494%` for treatment:
the scalar contract changes quality by `−12.427 pp`. The treatment also
learns the seen depths slightly less completely (`98.633%/99.023%` versus
`100%/100%`) and worsens both held-out depths in both seeds. The larger
evaluation removes the possibility that the rejection is due to the small
256-example batch.

## Decision

**V0.268 is rejected for adoption.** A normalized scalar contract and direct
query/read injection are not the missing reusable state interface: they
interfere with the learned circuit path and reduce depth transfer even on the
small `0..7` task. Do not add more scalar-loss weights, scalar scales, or
scalar authority sweeps, and do not scale this lane to 300M+ training as a
quality fix.

This does not reject a typed state idea in general. It closes this particular
one-dimensional affine/product packet. The next meaningful architecture must
use a genuinely vector-valued typed register with explicit operation/carry
slots, while keeping the sparse circuit path in the loop and measuring stage
state quality separately from final output accuracy.

Raw reports:

- `results/runs/v0_268_small07_control_seed17_2000.json`
- `results/runs/v0_268_small07_control_seed18_2000.json`
- `results/runs/v0_268_small07_structured_contract_seed17_2000.json`
- `results/runs/v0_268_small07_structured_contract_seed18_2000.json`
- `results/runs/v0_268_small07_paired_large_eval.json`

Evaluation helper: `benchmark_prior_free_checkpoint_eval.py`.
