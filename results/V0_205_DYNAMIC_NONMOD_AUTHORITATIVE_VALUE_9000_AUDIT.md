# V0.205 Authoritative Value Packet — 9k Stress Audit

## Question

V0.202–V0.203 showed that the authoritative scalar packet learned too
slowly at 1,000 steps, even with a direct contract loss. Does giving the same
packet a full 9,000-step budget recover the held-out depth-4 signal on the
corrected non-modular `0--7` task?

## Protocol

The corrected V0.202 configuration was trained from scratch for 9,000 CUDA
steps with seeds 17 and 18. The task uses ordinary integer arithmetic
(`modulus: null`), operands `0--7`, train depths `1--2`, held-out depths
`3--4`, `target_offset: 4096`, and a factorized exact `32,768`-class output.
The learned scalar packet is authoritative for the recurrent query and step
output; the dense writer and factorized circuit bank remain present as the
auxiliary path. No contract loss, extra capacity, or router change was added.

The first launch without `--heldout-depths` was stopped immediately by the
existing target-range validation (`min=-2,601,584`, `max=181,837,696` against
`num_classes=32,768`). It produced no result and is excluded. The results
below use the corrected train/held-out split and explicit `0--7` range flags.

## Results

| seed | train accuracy | held-out accuracy | depth 3 | depth 4 | train loss | held-out CE |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 91.99% | 66.99% | 72.07% | 61.91% | 0.4668 | 5.2268 |
| 18 | 87.60% | 64.45% | 69.34% | 59.57% | 0.4892 | 4.0743 |
| **mean** | **89.79%** | **65.72%** | **70.70%** | **60.74%** | — | — |

Both runs used `7,352,851` total parameters and an estimated `2,053,692`
active path. Evaluation used all `154` factor rows in seed17 and `153/154`
in seed18, so the result is not explained by a dead factor bank. Training
time was about `633` seconds per seed on CUDA.

## Interpretation

The longer budget improves learning compared with the V0.202 1,000-step
authoritative screen (`5.08%` held-out accuracy), but it does not recover the
usable non-authoritative factorized reference (`71.97%` mean at 3,000 steps).
The comparison is directional rather than a strict matched control because
the authoritative packet changes the state path. Crucially, increasing time
does not turn the authoritative packet into a reliable depth-transfer
mechanism: seen-depth fit remains incomplete and depth-4 stays near `60%`.

The failure is not primarily router starvation. The packet's algebraic
transition is still a learned scalar-to-state projection, and its authority
removes the dense state representation from the later operation. This makes
the current scalar-packet family unsuitable as the next scaled architecture.

## Decision

**REJECTED FOR ADOPTION AND SCALING.** Do not move this authoritative packet
to 500M/700M/1B and do not spend more steps on the same variant. Keep the
checkpoint and JSON as a negative control. P-004 remains open, but the next
experiment must use a genuinely different value/state representation or an
explicit structured transition rather than another scalar injection,
authority, contract weight, or simple residual update.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_7_factorized_authoritative_value.yaml`
- `results/runs/nonmod_depth4_values0_7_factorized_authoritative_seed17_9000.json`
- `results/runs/nonmod_depth4_values0_7_factorized_authoritative_seed18_9000.json`
- `results/checkpoints/nonmod_depth4_values0_7_factorized_authoritative_seed17_9000.pt`
- `results/checkpoints/nonmod_depth4_values0_7_factorized_authoritative_seed18_9000.pt`

## Reproduction

```powershell
python -u train_dynamic_composition.py `
  --config configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_7_factorized_authoritative_value.yaml `
  --steps 9000 --seed 17 `
  --run-id nonmod_depth4_values0_7_factorized_authoritative_seed17_9000 `
  --checkpoint results/checkpoints/nonmod_depth4_values0_7_factorized_authoritative_seed17_9000.pt `
  --output results/runs --examples-per-depth 512 --log-every 500 `
  --heldout-depths --train-value-min 0 --train-value-max 7 `
  --eval-value-min 0 --eval-value-max 7
```

Replace `17` with `18` in the seed and run-id/checkpoint names for the second
replication.
