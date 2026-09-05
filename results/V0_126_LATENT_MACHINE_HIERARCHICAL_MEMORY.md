# V0.126 `taklif22` Hierarchical Learned Memory

## Question

Can hierarchical addressing improve the 512-entity task-only memory failure
without changing the one-active-cell computation route?

## Protocol

The global learned memory from V0.125 is replaced by two differentiable
softmax stages: 32 learned buckets, then a learned row choice inside each
bucket. With 512 entities this reduces local row competition from 512 to 16.
No explicit memory-address loss is used. The task is the same bounded
two-dimensional vector program, with four top-1 MLP cells and labels-free
route consistency/balance.

```powershell
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 512 --memory-mode hierarchical --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2026 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_hierarchical_memory_nosup_512e_seed2026.json
python -u experiment_latent_computational_machine.py --device cuda --task-variant bounded-vector --value-dim 2 --num-entities 512 --memory-mode hierarchical --memory-supervision-weight 0.0 --structured-value-lane --structured-cell-kind mlp --steps 4000 --batch-size 128 --train-max-steps 4 --eval-min-steps 5 --eval-max-steps 6 --eval-batches 16 --log-every 1000 --seed 2027 --role-loss-weight 0.0 --state-supervision-weight 1.0 --route-consistency-weight 1.0 --route-balance-weight 1.0 --output results/runs/latent_computational_machine_v1_bounded_vector_hierarchical_memory_nosup_512e_seed2027.json
```

## Results

| seed | depth-6 MSE | fact-table B MSE | zero-memory MSE | memory accuracy | route usage range | decision |
|---:|---:|---:|---:|---:|---:|:---:|
| 2026 | **7.31e-5** | 1.39e-4 | 3.31e-4 | **92.3%** | 24.7--25.5% | partial pass |
| 2027 | **7.36e-5** | 9.23e-5 | 2.90e-4 | **94.9%** | 24.6--25.5% | partial pass |

Global 512-way task-only addressing gave depth-6 MSE `2.89e-4` and
`2.41e-4`, with retrieval `40.9%` and `42.6%`. Hierarchical addressing
therefore improves depth-6 error by roughly 3.3--3.9x and raises retrieval by
about 50 percentage points, without route collapse. It remains below the
near-perfect 32/128 and address-supervised 512 gates.

## Interpretation

The failure at 512 is not caused by the active computation cells: route usage
stays balanced and the same four-cell vector controller is used. Hierarchy
provides a real optimization benefit by narrowing the local memory choice,
but the fixed 32-bucket split still leaves a non-trivial bucket-learning
problem. The next experiment should expose the bucket count and compare 64
and 128 buckets, preserving the same task-only objective.

## Decision

**Keep as a promising architecture path, but do not promote as solved.**
Hierarchical memory is a better direction than global row-wise addressing;
continue with bucket-count ablations before Qwen transfer.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_hierarchical_memory_nosup_512e_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_hierarchical_memory_nosup_512e_seed2027.json`
