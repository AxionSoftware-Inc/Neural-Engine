# V0.24 three-seed validation of capacity growth

V0.23 found a positive signal from growing a 500M circuit bank out of a
trained 300M parent. V0.24 adds a third seed using the same parent, route
census, clone noise, 1,000-step parent-routing warm-up, and 5,000 total target
steps. The purpose is to test whether the improvement survives a change in
the training stream and child-row initialization noise.

The target remains 505.832M parameters with 38,600 circuits. The parent is the
299.544M `ne300m_lazy_quality_5000.pt` checkpoint. The third-seed run uses
`configs/ne_500m_growth_lazy_seed19.yaml` and its exact reproduction commands
are recorded in `results/V0_23_CAPACITY_GROWTH.md` with seed 19 substituted.

## Results

| Run | Full accuracy | Held-out accuracy | Avg steps | Dead circuits | Used circuits | Train time | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Growth, seed 17 | 50.55% | 51.67% | 1.326 | 51.44% | 18,744 | 509.9 s | 9,031 MiB |
| Growth, seed 18 | 51.85% | 53.02% | 1.288 | 55.01% | 17,366 | 527.1 s | 8,880 MiB |
| Growth, seed 19 | 51.02% | 51.04% | 1.301 | 54.27% | 17,650 | 509.5 s | 9,247 MiB |
| Mean ± population SD | **51.14% ± 0.54** | **51.91% ± 0.83** | — | 53.57% | — | — | — |

The scratch 500M reference reached 44.32% full and 45.63% held-out accuracy.
Thus the three-seed growth mean is +6.82 points full and +6.28 points
held-out over the scratch reference. The effect is consistent in direction
across all three seeds, although the grown 500M model still does not exceed
the 300M parent’s 56.80% full accuracy.

The third seed also restored multi-step behavior: depth-2 accuracy was 12.63%
and depth-3 accuracy 3.52%, versus 3.65% and 1.95% in the scratch 500M
reference. This remains a partial recovery, not a solved reasoning result.

## Decision

`taklif.md`’s parent-based capacity growth is a **stronger positive signal**
than random capacity expansion and should be retained as the default way to
prototype larger banks. It is not yet proof of fair 300M→500M capability
scaling because the target receives a trained parent and the extra training
budget is not matched to scratch training.

The next falsification is a composition/generalization benchmark, followed by
an independently trained parent or a larger growth step only if the
composition signal persists.
