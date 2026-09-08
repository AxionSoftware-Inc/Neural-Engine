# V0.204 Dense State Residual Update Audit

## Question

The state trace showed intermediate information decaying with depth. Does an
explicit residual update, `new_state = old_state + alpha * proposal`, preserve
that information better than the current overwrite writer?

## Protocol

The factorized-output 0--7 non-modular depth-4 configuration is unchanged
except for `state_update_mode: residual` and `state_residual_scale: 0.25`.
Training uses depths `1--2`, evaluation uses held-out depths `3--4`, batch size
`512`, and the same CUDA schedule. The router, circuit bank, output head, and
loss are unchanged.

## Results

| run | steps | train accuracy | held-out accuracy | depth 3 | depth 4 | final train loss |
|---|---:|---:|---:|---:|---:|---:|
| residual seed17 screen | 1,000 | 100.00% | 65.63% | 78.13% | 53.13% | 0.00156 |
| residual seed17 full | 3,000 | 100.00% | 66.99% | 80.47% | 53.52% | 0.00025 |
| factorized control seed17 | 3,000 | 100.00% | 71.09% | 82.81% | 59.38% | 0.00367 |

The residual path reaches the seen-depth target but loses `4.10 pp` at 1,000
steps and `4.10 pp` at 3,000 steps against the same seed's factorized control.
Depth-4 loses `5.86 pp` at the final screen.

## Decision

**REJECTED FOR ADOPTION.** Preserving the old dense state by additive residual
accumulation does not preserve the useful compositional representation. The
proposal itself contains route/circuit interference, so accumulation carries
error forward as well as signal. No second seed is warranted after the first
full screen misses the quality gate by more than four points.

This closes the simple scalar and residual state-preservation family for the
current 0--7 task. It does not mean recurrent architectures are impossible;
it means the current dense writer needs an explicitly structured transition or
different state representation, not another additive scale.

## Next action

Do not increase capacity, candidate pool, or residual scale. The next useful
experiment should separate operation computation from persistent value storage
with a genuinely different state representation and a matched control, or
return to the Qwen/FFN circuit-transplant lane. The 0--3 typed-write model and
the flat 0--7 model remain the current reference checkpoints.

## Artifacts

- `configs/ne_dynamic_300m_nonmod_depth4_typed_write_adapter_values0_7_factorized_state_residual.yaml`
- `neural_engine/dynamic_register.py` (`state_update_mode`)
- `results/runs/nonmod_depth4_values0_7_factorized_state_residual_screen_seed17_1000.json`
- `results/runs/nonmod_depth4_values0_7_factorized_state_residual_seed17_3000.json`
