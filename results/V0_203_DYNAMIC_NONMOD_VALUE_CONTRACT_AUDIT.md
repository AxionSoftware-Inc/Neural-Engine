# V0.203 Authoritative Value Contract Audit

## Question

The authoritative scalar packet did not learn the seen depths. Can a direct
normalized value-contract objective teach that packet to carry the correct
intermediate accumulator?

## Protocol

The V0.202 authoritative scalar configuration is kept fixed on the ordinary
integer `0--7` task: train depths `1--2`, held-out depths `3--4`, target offset
`4096`, factorized `32,768`-class output, batch size `512`, and CUDA. The
contract loss is a smooth-L1 loss between the scalar packet and raw stage
targets, divided by a target scale. Two short seed17 screens are compared:
weight `1.0`, scale `64`, and weight `100.0`, scale `64`.

## Results

| contract weight | scale | steps | train accuracy | held-out accuracy | final train loss |
|---:|---:|---:|---:|---:|---:|
| 0 (authoritative control) | — | 1,000 | 27.73% | 5.08% | 3.59 |
| 1 | 64 | 1,000 | 27.34% | 5.08% | 3.60 |
| 100 | 64 | 1,000 | 26.56% | 6.25% | 4.42 |

The weight-1 contract is too weak to change the trajectory. The weight-100
contract dominates optimization but still does not learn the seen task.

## Decision

**REJECTED.** Direct scalar contract supervision does not rescue the
authoritative packet. This closes the current scalar-packet family: query-only
injection, persistent read injection, authoritative use, and direct contract
loss have all failed to produce a reliable quality result. No capacity scaling
is justified from these tests.

The remaining useful reference is the non-authoritative typed-write path, where
the scalar lane may remain as an opt-in diagnostic. The next architecture test
returns to the dense state writer and tests an explicit residual state update,
which preserves the prior state instead of replacing it through a bounded
writer at every operation.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_7_factorized_authoritative_value_contract.yaml`
- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_7_factorized_authoritative_value_contract_strong.yaml`
- `train_dynamic_composition.py` (normalized scalar contract loss)
- `results/runs/nonmod_depth4_values0_7_factorized_authoritative_value_contract_screen_seed17_1000.json`
- `results/runs/nonmod_depth4_values0_7_factorized_authoritative_value_contract_scale64_screen_seed17_1000.json`
- `results/runs/nonmod_depth4_values0_7_factorized_authoritative_value_contract_strong_screen_seed17_1000.json`
