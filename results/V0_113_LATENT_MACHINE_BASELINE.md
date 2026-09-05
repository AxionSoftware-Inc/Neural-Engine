# V0.113 `taklif22` Latent Machine Baseline

## Question

Can the minimal four-part machine in `taklif22` learn reusable computation
while mutable facts remain outside the computation cells?

## Protocol

The first V1 diagnostic uses an external 32-entry fact table, an 8-slot
working memory, a 32-dimensional latent instruction, four computation cells,
and hard top-1 execution at evaluation. The controller sees the current state,
pooled working memory, and an operation token; the cells do not receive entity
identity or fact-table identity. Training uses 2--4-step programs. Evaluation
uses unseen 5--6-step compositions, a fact-table swap, and a zero-memory
intervention. A small role-loss is enabled only as the synthetic diagnostic
alignment stage described by `taklif22`.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --steps 4000 --log-every 500 --output results/runs/latent_computational_machine_v1_seed2026.json
```

Hardware: NVIDIA GeForce RTX 3060 12GB, driver 591.86.

## Results

The model has 92,545 stored parameters. One active cell contributes 8,448
parameters, or 9.14% of the total; one cell is selected per active program
step. Training task loss reached 0.0030 and the diagnostic route loss reached
1e-5. Cell usage stayed broad at roughly 24.6--25.5% per cell, so the failure
is not dead-cell collapse.

| evaluation | MSE | MAE | interpretation |
|---|---:|---:|---|
| in-range 2--4 steps | 0.0098 | 0.062 | short-program fit is good |
| unseen 5--6 steps, fact table A | 62.83 | 0.284 | fails long-depth stability |
| unseen 5--6 steps, swapped fact table B | 8.04 | 0.154 | fact intervention remains finite but inherits depth failure |
| zero-memory intervention | 67.74 | 0.447 | memory path is measurably used |

Routes reuse the same selected cell for identical program tokens across facts
(same-program Jaccard 1.00), while different programs have lower overlap
(0.641). This is a routing signal, not proof of a good latent language.

## Decision

**The unnormalized minimal state machine is rejected for the variable-depth
gate.** It fits short training programs but its residual state drifts badly at
5--6 steps. The balanced route histogram and zero-memory degradation localize
the failure to state dynamics/normalization, not memory retrieval or dead
cells.

This is a useful falsification result for `taklif22`: adding cells or enlarging
the fact table would not address the measured failure. The next gate adds only
the prescribed bounded residual update and state LayerNorm, then repeats the
same task, fact swap, and depth controls.

## Artifact

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_fixed_seed2026.json`
