# V0.163 Qwen Learned Latent-Basis Audit

## Question

Can Qwen's distributed FFN function be re-expressed in a new learned latent
basis, so that only a small subset of basis decoders is active and the copied
neuron-slice bottleneck disappears?

This is the first direct test of `taklif15.md`. It is intentionally separate
from the copied-Qwen cross-group experiment in V0.161.

## Setup

The child does not copy Qwen FFN neuron weights. It learns one shared nonlinear
rank-384 feature map and 16 latent output decoders. The router selects 4 of 16
decoders per token (25% active decoder budget). Only the selected decoders are
used in the hard path. Four late Qwen layers (23--26) are replaced. Training
uses the diversified calibration corpus for 300 total steps, including 200
hard-route steps at learning rate `3e-4`; evaluation uses the independent
held-out corpus.

## Result

| Metric | Result |
|---|---:|
| Held-out CE delta | +0.1519 |
| Teacher top-1 agreement | 79.15% |
| Quality gate (`+0.05`) | FAIL |
| Child scalar/storage fraction | 0.764x parent |
| End-to-end timing | 18.65x parent |
| Approximate wall time | 55 minutes |

The learned basis reduces stored scalar parameters below the copied parent
FFN, but it does not preserve the teacher function. The soft training path
evaluates all 16 basis decoders, which explains the extreme training and
prototype runtime cost; this is not a deployment-speed claim. The quality
failure remains even before that implementation issue is solved.

## Decision

**NO-GO for the current randomly initialized shared-basis formulation.** The
result is materially worse than the validated 4-layer, 50%-active cross-group
child (`+0.0181`/`+0.0155`) and misses the 25%-active target by a wide margin.
Do not keep increasing latent rank or basis count with the same initialization
and soft-all-bases training path.

The useful lesson is narrower: a factorized basis can reduce physical storage,
but storage compression alone does not recover Qwen capability. A next basis
experiment is justified only if it is teacher-derived from Qwen activations or
down-projection structure, and it must avoid evaluating the full basis during
training. Otherwise keep the copied cross-group path as the local quality
control and move to a different representation.

## Artifact

- `benchmark_qwen_multi_layer_transplant.py`
- `results/runs/qwen_multi_layer_latent_basis_4layers_b16k4_rank384_diverse_eval_hard200_lr3e-4_b8s128_seed2026.json`
