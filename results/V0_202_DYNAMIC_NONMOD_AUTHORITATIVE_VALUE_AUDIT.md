# V0.202 Authoritative Value Packet Learning Audit

## Question

Can a learned scalar value packet become the authoritative recurrent state,
with the dense accumulator retained only as an auxiliary circuit path?

## Protocol

The V0.201 factorized-output control is modified so the operation-conditioned
bilinear scalar state is used as the next operation's read state and as the
step output state. Its learned transition receives `(old, operand,
old*operand, bias)`. The task is ordinary integer arithmetic on values `0--7`,
train depths `1--2`, held-out depths `3--4`, target offset `4096`, batch size
`512`, and CUDA. No bank or router capacity is changed.

## Results

The 20-step smoke test already showed no immediate failure, but learning was
far slower than the factorized control. A 1,000-step seed17 screen reached:

| steps | train accuracy | held-out accuracy | train loss |
|---:|---:|---:|---:|
| 20 | 0.00% | 0.00% | 9.74 |
| 500 | — | — | 4.44 |
| 1,000 | 27.73% | 5.08% | 3.59 |

Continuing the same seed to 1,500 steps left the loss at `3.61`, with no
sign of the earlier variants' near-zero training loss. The run was stopped
before 3,000 steps because the seen-depth learning gate had already failed.

## Decision

**REJECTED AS A LEARNING CONFIGURATION.** Making the scalar lane authoritative
without directly supervising its numeric contract prevents the model from
learning the seen task in the available budget. This does not prove that a
persistent value packet is impossible; it shows that authority must be
introduced with an explicit, normalized value-contract objective or a more
expressive packet transition.

Do not scale this variant and do not interpret its held-out result as a
capacity result. The dense writer remains the usable reference path.

## Next action

If the persistent-register hypothesis is kept alive, the next test will add a
small normalized scalar-contract loss directly to the learned packet, while
keeping the output and routing paths fixed. The loss must be reported
separately from final CE and exact accuracy. If that still fails to learn the
seen depths, close the scalar-packet branch and move to a different state
representation rather than adding more capacity.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_7_factorized_authoritative_value.yaml`
- `neural_engine/dynamic_register.py` (`structured_scalar_authoritative`)
- `results/runs/nonmod_depth4_values0_7_factorized_authoritative_value_screen_seed17_1000.json`
