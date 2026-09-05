# V0.147 Qwen Two-Layer Calibration Rank 8

## Question

Can a very small learned interface correction stabilize two adjacent sparse
children without relying on expensive joint refinement or a large capacity
increase?

## Protocol

Layers 25 and 26 of Qwen3-0.6B use normless four-expert top-2 grouped banks.
Each bank is wrapped in the existing zero-start `CalibratedChild` residual with
rank 8: a low-rank correction is applied to the bank output, while the main
computation still executes only two of four expert bodies per token. Each child
receives 300 local FFN distillation steps; no joint full-model refinement is
used. The quality gate is the combined `shared_alpha_0` path on 8×128 held-out
batches.

## Results

| seed | combined alpha=0 CE delta | combined top-1 agreement | child params | parameter fraction | quality gate |
|---:|---:|---:|---:|---:|:---:|
| 2026 | +0.0184 | 96.53% | 3,299,460 | 34.96% | pass |
| 2027 | +0.0289 | 96.66% | 3,299,460 | 34.96% | pass |

The uncalibrated normless grouped seed-2026 control was `+0.0993` and failed;
rank 32 reached `+0.0404` and `−0.0001` across two seeds. Rank 8 therefore
retains the improvement with 16,384 extra parameters per child, versus 65,536
extra parameters at rank 32. The active expert-body fraction remains 50%.

## Interpretation

The composition error is best treated as a small interface mismatch rather
than solved by multiplying expert capacity. A zero-start low-rank correction
lets each child preserve its sparse local function while learning the residual
needed by the surrounding Qwen representation. Rank 8 is already sufficient
on both seeds in this gate.

## Decision

**Accept rank-8 calibration as the current two-layer quality default.** Before
testing a larger teacher or more layers, add a two-layer timing measurement and
verify that the low-rank correction does not erase the V0.144 FFN speedup. Keep
rank 32 as a robustness control, not the default.

## Artifacts

- `benchmark_qwen_two_layer_transplant.py`
- `results/runs/qwen_two_layer_routed_grouped_nonorm_calrank8_b8s128_seed2026.json`
- `results/runs/qwen_two_layer_routed_grouped_nonorm_calrank8_b8s128_seed2027.json`
