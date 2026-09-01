# V0.23 capacity growth from a trained parent

V0.22 showed that a larger random circuit bank did not convert stored
capacity into useful quality. V0.23 tests the growth strategy proposed in
`taklif.md`: initialize new circuit rows from heavily used parent rows, keep
the parent routing geometry during a warm-up, and expose the expanded bank
only after the new model starts from a useful state.

This is a warm-start capability-conversion experiment, not a scratch-trained
500M versus scratch-trained 300M comparison. The parent is the trained
299.544M checkpoint `ne300m_lazy_quality_5000.pt`. The target has 505.832M
parameters, 22,800 parent circuits, and 38,600 total circuits.

## Method

1. Run a 64-batch route census over the parent and rank selected circuits.
2. Copy all parent controller, router, and circuit-prefix weights exactly.
3. Initialize 15,800 child rows by cloning the most-used parent rows with
   0.05 relative Gaussian noise.
4. Train the target for 1,000 steps with the parent’s five-level tree and
   22,800-row routing capacity.
5. Expose all 38,600 rows and the sixth tree level, then continue to 5,000
   total steps.

The growth initializer was checked before training: on the same batch,
parent and target warm-up logits had `max_logit_delta = 0.0` and their selected
routes were identical. An earlier implementation failed to copy the parent
prefix of the differently shaped bank tensors; that run was discarded and is
not included in the results below.

## Reproduction

```powershell
python grow_capacity.py --parent-checkpoint results/checkpoints/ne300m_lazy_quality_5000.pt --target-config configs/ne_500m_growth_lazy.yaml --output results/checkpoints/ne500m_growth_init.pt --device cuda --count-batches 64 --count-batch-size 128 --clone-noise 0.05
python train.py --config configs/ne_500m_growth_lazy.yaml --init-checkpoint results/checkpoints/ne500m_growth_init.pt --steps 5000 --device cuda --balanced-train --run-id ne500m_growth_lazy_quality_5000_v2 --output results/runs --log-every 500 --checkpoint results/checkpoints/ne500m_growth_lazy_quality_5000_v2.pt

python grow_capacity.py --parent-checkpoint results/checkpoints/ne300m_lazy_quality_5000.pt --target-config configs/ne_500m_growth_lazy_seed18.yaml --output results/checkpoints/ne500m_growth_init_seed18.pt --device cuda --count-batches 64 --count-batch-size 128 --clone-noise 0.05
python train.py --config configs/ne_500m_growth_lazy_seed18.yaml --init-checkpoint results/checkpoints/ne500m_growth_init_seed18.pt --steps 5000 --device cuda --balanced-train --run-id ne500m_growth_lazy_seed18_5000 --output results/runs --log-every 500 --checkpoint results/checkpoints/ne500m_growth_lazy_seed18_5000.pt
```

Held-out and active-budget evaluation:

```powershell
python analyze_active_budget.py --checkpoint results/checkpoints/ne500m_growth_lazy_quality_5000_v2.pt --device cuda --active-circuits 4 8 16 --examples-per-task 64 --batch-size 128 --iterations 50 --output results/runs/ne500m_growth_lazy_quality_active_budget_v2.json
python analyze_active_budget.py --checkpoint results/checkpoints/ne500m_growth_lazy_seed18_5000.pt --device cuda --active-circuits 8 --examples-per-task 64 --batch-size 128 --iterations 50 --output results/runs/ne500m_growth_lazy_seed18_active_budget.json
```

## Results

| Run | Total params | Full accuracy | Held-out accuracy | Avg steps | Depth-2 | Depth-3 | Dead circuits | Used circuits | Train time | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500M scratch, no coverage | 505.832M | 44.32% | 45.63% | 1.000 | 3.65% | 1.95% | 67.49% | 12,548 | 509.6 s | 9,547 MiB |
| 500M growth, seed 17 | 505.832M | **50.55%** | **51.67%** | 1.326 | 11.85% | 4.56% | **51.44%** | 18,744 | 509.9 s | 9,031 MiB |
| 500M growth, seed 18 | 505.832M | **51.85%** | **53.02%** | 1.288 | 15.76% | 5.47% | 55.01% | 17,366 | 527.1 s | 8,880 MiB |
| Growth mean | 505.832M | **51.20%** | **52.34%** | — | — | — | 53.23% | — | — | — |

The growth mean improves scratch 500M held-out accuracy by 6.72 percentage
points and full accuracy by 6.88 points. The two seeds agree on the direction
of the result. The 300M parent remains stronger at 56.80% full accuracy, so
growth is a positive capacity-conversion signal, not yet proof that 500M is
the best final model.

The active-budget sweep on seed 17 found 50.83%, 50.83%, and 51.15% full
accuracy for k=4, 8, and 16, with held-out accuracy 51.67%, 51.67%, and
51.56%. Throughput was 13,866, 12,133, and 11,889 samples/s. The default k=8
remains the best balance; increasing active circuits does not explain the
growth improvement.

## Interpretation and next step

The negative scratch result was caused mainly by a cold, under-trained
expanded bank and routing/depth collapse. Parent-based growth keeps useful
operators alive while adding new capacity, and it restores part of the
depth-2/3 trajectory. This is the first positive result for the
`taklif.md` capacity-growth proposal.

The result is still not a fully controlled scratch A/B because the growth
model receives 5,000 steps of prior 300M training. The next falsification is
to repeat growth from an independently trained 300M parent or run a third
seed, then test whether the benefit persists on the composition benchmark.
