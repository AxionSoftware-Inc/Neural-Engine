# V0.27 curriculum and audited composition scaling

V0.27 separates training exposure from stored capacity. A 300M model is
first trained on all nine operation pairs for 10,000 steps, then fine-tuned
for 5,000 steps with the two operation orders hidden. The same parent-based
growth procedure is then used to produce a 500M target. The route-exploration
probability is 5% during training only.

The earlier composition reports used 64 random examples per pair. V0.27 also
uses the deterministic evaluator from `evaluate_composition.py`: every model
is tested on all `16^3 = 4,096` operand triples for each hidden pair. These
grid results are the primary quality numbers below; the small random-batch
numbers should not be used as a scaling claim.

The seed-19 configuration was recorded in commit `5c42739`. The main training
commands were:

```powershell
python train_composition.py --config configs/ne_300m_composition_explore_all.yaml --steps 10000 --device cuda --run-id ne300m_composition_explore_all_10000 --output results/runs --examples-per-task 64 --log-every 1000 --checkpoint results/checkpoints/ne300m_composition_explore_all_10000.pt
python train_composition.py --config configs/ne_300m_composition_explore_seed18.yaml --init-checkpoint results/checkpoints/ne300m_composition_explore_all_10000.pt --steps 5000 --device cuda --run-id ne300m_composition_curriculum_hidden_5000 --output results/runs --examples-per-task 512 --log-every 500 --checkpoint results/checkpoints/ne300m_composition_curriculum_hidden_5000.pt
python train_composition.py --config configs/ne_300m_composition_explore_all_seed19.yaml --steps 10000 --device cuda --run-id ne300m_composition_explore_all_seed19_10000 --output results/runs --examples-per-task 64 --log-every 1000 --checkpoint results/checkpoints/ne300m_composition_explore_all_seed19_10000.pt
python train_composition.py --config configs/ne_300m_composition_explore_seed19.yaml --init-checkpoint results/checkpoints/ne300m_composition_explore_all_seed19_10000.pt --steps 5000 --device cuda --run-id ne300m_composition_curriculum_hidden_seed19_5000 --output results/runs --examples-per-task 512 --log-every 500 --checkpoint results/checkpoints/ne300m_composition_curriculum_hidden_seed19_5000.pt
python grow_capacity.py --parent-checkpoint results/checkpoints/ne300m_composition_curriculum_hidden_seed19_5000.pt --target-config configs/ne_500m_composition_growth_explore_seed19.yaml --output results/checkpoints/ne500m_composition_curriculum_growth_seed19_init.pt --device cuda --census-mode composition --count-batches 64 --count-batch-size 128 --clone-noise 0.05
python train_composition.py --config configs/ne_500m_composition_growth_explore_seed19.yaml --init-checkpoint results/checkpoints/ne500m_composition_curriculum_growth_seed19_init.pt --steps 5000 --device cuda --run-id ne500m_composition_curriculum_growth_seed19_5000 --output results/runs --examples-per-task 512 --log-every 500 --checkpoint results/checkpoints/ne500m_composition_curriculum_growth_seed19_5000.pt
```

## Deterministic hidden-pair results

| Model | Params | Seed | Hidden-pair grid accuracy |
|---|---:|---:|---:|
| NE-20M prior reference | 20.37M | 17 | 12.40% |
| NE-300M curriculum | 299.66M | 18 | 12.83% |
| NE-300M curriculum | 299.66M | 19 | 14.76% |
| NE-300M curriculum mean ± population SD | 299.66M | 18/19 | **13.79% ± 0.96** |
| NE-500M curriculum growth | 505.95M | 17 | 12.54% |
| NE-500M curriculum growth | 505.95M | 19 | 13.01% |
| NE-500M curriculum growth mean ± population SD | 505.95M | 17/19 | **12.77% ± 0.24** |

The curriculum materially improves the 300M result compared with the earlier
5,000-step hidden-only runs. However, growing from 300M to 500M does not add
quality: the two-seed mean drops by 1.02 points. The 500M model remains
hardware-feasible on the 12 GiB RTX 3060, but the extra stored circuits are not
yet converting into additional compositional capability.

The 20M all-pairs 20,000-step control reaches 31.18% on the same deterministic
all-pairs grid, showing that optimization/training exposure matters strongly.
It does not by itself establish hidden-pair generalization, so it is recorded
as a learnability control rather than a scaling result.

## Decision

**WEAK POSITIVE for curriculum; NO-GO for 700M/1B quality scaling.**

Keep the 300M curriculum recipe as the current quality baseline. Do not spend
the next budget on a 700M or 1B bank until a circuit-composition or working-
memory intervention makes the 500M growth model reliably exceed the 300M
parent. The immediate research target is useful capacity conversion, not
larger capacity by itself.
