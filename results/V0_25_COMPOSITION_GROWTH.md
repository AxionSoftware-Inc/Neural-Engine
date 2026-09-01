# V0.25 composition benchmark at 300M and 500M

V0.25 follows the three-seed growth validation with a harder falsification:
the same hidden operation-order benchmark is trained at 300M and 500M. The
benchmark hides `add -> multiply` and `multiply -> add` during training and
tests them separately. All runs use the shared numeric/Fourier frontend,
LazyAdamW, seed 17, 5,000 steps, and the same evaluation protocol.

The benchmark code and configurations were recorded in commit `72a6c9e`
(`Add composition capacity growth benchmark`). The exact commands were:

```powershell
python train_composition.py --config configs/ne_300m_composition_parent.yaml --steps 5000 --device cuda --run-id ne300m_composition_parent_5000 --output results/runs --log-every 500 --checkpoint results/checkpoints/ne300m_composition_parent_5000.pt
python train_composition.py --config configs/ne_300m_composition_active16.yaml --steps 5000 --device cuda --run-id ne300m_composition_active16_5000 --output results/runs --log-every 500 --checkpoint results/checkpoints/ne300m_composition_active16_5000.pt
python train_composition.py --config configs/ne_300m_composition_stage.yaml --steps 5000 --device cuda --run-id ne300m_composition_stage_5000 --output results/runs --log-every 500 --checkpoint results/checkpoints/ne300m_composition_stage_5000.pt
python grow_capacity.py --parent-checkpoint results/checkpoints/ne300m_composition_parent_5000.pt --target-config configs/ne_500m_composition_growth.yaml --output results/checkpoints/ne500m_composition_growth_init.pt --device cuda --census-mode composition --count-batches 64 --count-batch-size 128 --clone-noise 0.05
python train_composition.py --config configs/ne_500m_composition_scratch.yaml --steps 5000 --device cuda --run-id ne500m_composition_scratch_5000 --output results/runs --log-every 500 --checkpoint results/checkpoints/ne500m_composition_scratch_5000.pt
python train_composition.py --config configs/ne_500m_composition_growth.yaml --init-checkpoint results/checkpoints/ne500m_composition_growth_init.pt --steps 5000 --device cuda --run-id ne500m_composition_growth_5000 --output results/runs --log-every 500 --checkpoint results/checkpoints/ne500m_composition_growth_5000.pt
```

## Results

| Run | Total params | Train-split accuracy | Held-out accuracy | Active circuits | Train time |
|---|---:|---:|---:|---:|---:|
| NE-20M parent, prior reference | 20.37M | 20.09% | 12.50% | 8 | 175.0 s |
| NE-300M parent | 299.66M | 10.71% | 7.03% | 8 | 331.3 s |
| NE-300M active16 | 299.66M | 7.81% | 8.59% | 16 | 337.6 s |
| NE-300M + stage loss 0.1 | 299.66M | 11.16% | 9.38% | 8 | 338.6 s |
| NE-500M scratch | 505.95M | 8.48% | **10.16%** | 8 | 489.0 s |
| NE-500M parent-growth | 505.95M | 8.93% | 9.38% | 8 | 486.6 s |

The exact-match random baseline is 1.56% (1/64). All large models are above
that baseline, but they remain substantially underfit on this task. The 500M
scratch model beats the 300M parent by 3.13 held-out points, but remains below
the 20M parent by 2.34 points. Parent-growth does not beat 500M scratch in
this composition run and is 3.12 points below the 20M parent.

The apparent 300M active16 and stage-loss improvements are single-seed,
small absolute changes and do not establish a reliable scaling law. None of
the large models learned the hidden operation orders well enough to support a
generalization claim.

## Decision

**FAIL / NO-GO for 700M or 1B based on the current composition result.**

This is not evidence that the Neural Engine idea is useless. It is evidence
that the current sparse router/circuit training recipe does not convert extra
stored capacity into compositional skill. The earlier numeric benchmark still
shows a useful active-compute and throughput Pareto signal, and parent-growth
was positive on that easier task. However, the richer composition benchmark
is the more important gate for claiming capability scaling.

The next experiment should target credit assignment and circuit composition
before another size jump: a controlled circuit-family or route-exploration
ablation at 20M/300M, with the same hidden-pair benchmark. If that improves
the 300M result, the recipe can be re-tested at 500M. If it does not, the
working-state/circuit composition design needs revision. 500M remains a
useful systems milestone, but it is not yet a quality milestone.
