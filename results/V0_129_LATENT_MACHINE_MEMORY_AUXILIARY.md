# V0.129 `taklif22` Low-Weight Memory Address Guidance

## Question

Can a small auxiliary address objective recover the 512-entity hierarchical
memory without dominating the task/state objective?

## Protocol

This keeps V0.126's 512-entity, 32-bucket hierarchical memory and V0.121's
bounded two-dimensional task. Four computation cells remain top-1 and the
operation-role labels remain disabled. The only change is a small
`memory-supervision-weight=0.1`; the full address-supervised control uses
weight 1.0 and the task-only baseline uses weight 0.0.

## Results

| memory guidance | seed | depth-6 MSE | fact-table B MSE | in-range MSE | memory accuracy | decision |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.0 | 2026 | 7.31e-5 | 1.39e-4 | 7.34e-4 | 92.3% | partial |
| 0.0 | 2027 | 7.36e-5 | 9.23e-5 | 6.22e-4 | 94.9% | partial |
| 0.1 | 2026 | **8.46e-7** | **9.88e-7** | **2.91e-6** | **100%** | pass |
| 0.1 | 2027 | **6.46e-7** | **7.63e-7** | **3.88e-6** | **100%** | pass |
| 1.0 | 2026 | 7.45e-7 | 9.34e-7 | 3.13e-6 | 100% | control |

At weight `0.1`, route usage remains balanced and one computation cell is
active per step. The machine stores 94,313 parameters, while the active
vector cell has 162 parameters (`0.17%`). The low-weight objective recovers
exact memory addresses and reaches essentially the same task quality as the
full address-supervised control.

## Interpretation

The 512-row task-only failure was an optimization bottleneck, not a lack of
cell capacity. Hierarchy alone recovered most of the gap; a small address
signal completes the lookup without displacing the state/task objective. This
is the first stable 512-row configuration with exact memory retrieval and
labels-free computation routing across two seeds.

The address objective is still a form of memory supervision, so this should
not be described as purely emergent open-world memory. The learned query/key
bank is also tied to the fixed entity inventory. The next generalization gate
should use content-addressed or hierarchical keys with held-out entities,
then test longer programs and larger cell banks.

## Decision

**Promote the 0.1-guided hierarchical configuration as the current synthetic
reference.** Keep the task-only result as a meaningful ablation, but do not
transfer to Qwen yet; first remove the fixed-entity assumption and test
generalization to unseen addresses.

## Artifacts

- `experiment_latent_computational_machine.py`
- `results/runs/latent_computational_machine_v1_bounded_vector_hierarchical32_memory_weight01_512e_seed2026.json`
- `results/runs/latent_computational_machine_v1_bounded_vector_hierarchical32_memory_weight01_512e_seed2027.json`
