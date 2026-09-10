# P-007 native step-specific circuit adapter audit

Date: 2026-09-10  
Branch: `exp/track-runtime`

## Hypothesis

The selected circuit correction is shared across recurrent steps, even though
step-1, step-2, and step-3 computations have different roles. A small
step-specific low-rank residual adapter was added after the selected circuit
bank output:

```text
delta_step = delta + scale * GELU(delta @ down_step) @ up_step
```

The up projection is zero-initialized, so the default path is unchanged when
the feature is disabled. The rank-8 adapter adds 18,432 trainable parameters
to the 300M model and does not change route selection or active circuit count.

## Short paired screen

300M ordered shared-route-key bank, balanced training, 3,000 steps, seed17/18:

| Arm | Mean accuracy | Mean CE | Hard-task mean | Dead traffic |
|---|---:|---:|---:|---:|
| baseline, no adapter | 68.451% | 0.99639 | 33.366% | 40.17% |
| rank-8 adapter, all steps | 69.063% | 0.97968 | 34.668% | 41.52% |
| adapter, steps 2/3 only | 68.555% | 0.97838 | 33.529% | 38.16% |

The all-step adapter is the strongest arm. Disabling it at step 1 loses most
of the accuracy gain, so the improvement is not only a depth-2/3 correction;
the first step also needs a learned interface for the later state trajectory.

## Long-budget capacity check

The all-step rank-8 adapter was retrained from scratch for 10,000 steps.

| Scale | Baseline accuracy | Adapter accuracy | Accuracy delta | Baseline hard mean | Adapter hard mean | Hard delta | CE delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 300M | 77.096% | 77.448% | **+0.352 pp** | 45.247% | 46.517% | **+1.270 pp** | **−0.00220** |
| 500M | 77.292% | 77.318% | **+0.026 pp** | 46.940% | 47.233% | **+0.293 pp** | **−0.00317** |

The adapter improves hard-task accuracy and CE at both scales, but overall
accuracy does not scale from 300M to 500M (`−0.130 pp`). Thus it improves the
state/circuit interface without fully solving P-003 capacity scaling.

## Distribution probes

The same 10k checkpoints were evaluated on the valid native vocabulary. Delta
is adapter minus the corresponding 300M or 500M baseline under the same probe
protocol.

| Scale / probe | Accuracy delta | Hard-task delta | Dead-traffic delta |
|---|---:|---:|---:|
| 300M uniform | +0.469 pp | +1.302 pp | −3.16 pp |
| 300M combination holdout | +0.655 pp | +1.801 pp | −3.50 pp |
| 300M low edge `[0,7]` | +0.625 pp | +1.063 pp | −4.16 pp |
| 300M high edge `[56,63]` | **−1.389 pp** | +1.020 pp | −6.07 pp |
| 500M uniform | +0.100 pp | +0.510 pp | −0.32 pp |
| 500M combination holdout | +0.013 pp | +0.694 pp | −0.03 pp |
| 500M low edge `[0,7]` | −1.575 pp | +0.119 pp | −0.91 pp |
| 500M high edge `[56,63]` | −1.047 pp | −0.488 pp | −2.89 pp |

The adapter is therefore not yet distribution-robust: high-edge accuracy
regresses even while dead traffic falls. A separate numeric/value
representation issue remains open.

## Decision

**PROMISING OPT-IN; not default.** This is the first native intervention that
survives both seeds and 10k training with a measurable hard-task improvement.
It should be carried forward as the current leading candidate, but 700M/1B
should wait until the high-edge regression and 300M→500M accuracy plateau are
understood. The next test should target value-regime robustness or validate a
bounded step adapter with a held-out high-edge gate, not blindly increase the
bank.

## Reproduction

```powershell
python train.py --config configs/ne_300m_v12_factorized_shared_routekeys_step_adapter.yaml --steps 10000 --device auto --balanced-train --seed 17 --run-id ne300_step_adapter_s17_10000 --output results/runs --checkpoint results/checkpoints/ne300_step_adapter_s17_10000.pt
python train.py --config configs/ne_300m_v12_factorized_shared_routekeys_step_adapter.yaml --steps 10000 --device auto --balanced-train --seed 18 --run-id ne300_step_adapter_s18_10000 --output results/runs --checkpoint results/checkpoints/ne300_step_adapter_s18_10000.pt
python train.py --config configs/ne_500m_v12_factorized_shared_routekeys_step_adapter.yaml --steps 10000 --device auto --balanced-train --seed 17 --run-id ne500_step_adapter_s17_10000 --output results/runs --checkpoint results/checkpoints/ne500_step_adapter_s17_10000.pt
python train.py --config configs/ne_500m_v12_factorized_shared_routekeys_step_adapter.yaml --steps 10000 --device auto --balanced-train --seed 18 --run-id ne500_step_adapter_s18_10000 --output results/runs --checkpoint results/checkpoints/ne500_step_adapter_s18_10000.pt
python audit_native_ood.py --checkpoints results/checkpoints/ne300_step_adapter_s17_10000.pt results/checkpoints/ne300_step_adapter_s18_10000.pt --batches 24 --device auto --output results/diagnostic_native_step_adapter_ood_10000.json
python audit_native_ood.py --checkpoints results/checkpoints/ne500_step_adapter_s17_10000.pt results/checkpoints/ne500_step_adapter_s18_10000.pt --batches 24 --device auto --output results/diagnostic_native_step_adapter_500m_ood_10000.json
```

