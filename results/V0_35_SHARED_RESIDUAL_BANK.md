# V0.35 shared-residual circuit bank

Status: **rejected; the shared residual alone did not fix routing**

## Change

V0.35 kept the global hierarchical router and added a shared low-rank residual
path to every circuit-bank stage. Selected circuit rows remained sparse
adapters. The intent was to provide a common primitive computation even when
the global route selected a weak or under-trained row.

This remained a typed-register, non-attention architecture:

```text
(a,b,op1) -> partial -> (partial,c,op2) -> final -> readout
```

## Exact run

```text
python train_composition.py --config configs/ne_typed_register_20m_shared_all.yaml --steps 10000 --device cuda --run-id ne_typed_register_20m_shared_all_10000 --output results/runs --examples-per-task 512 --log-every 1000 --checkpoint results/checkpoints/ne_typed_register_20m_shared_all_10000.pt
```

Hardware: NVIDIA GeForce RTX 3060 12 GiB. Seed: 17. Training time: 614.4 s.
The run used 19,811,841 parameters and an estimated 1,542,657 active
parameters.

## Full-domain results

| Model | All nine pairs | Hidden pairs after 5k adaptation |
|---|---:|---:|
| 20M shared residual | 57.61% | 60.49% |
| Existing 20M reference | 58.10% | 64.89% |

The shared path did not recover the all-pairs reference and made the hidden
result worse. The common transform received dense gradient, but it did not
teach the sparse rows how to specialize or make the route more stable.

## Decision

Reject as a standalone scaling fix. The next experiment must change how circuit
capacity is represented or addressed, rather than merely adding one common
residual computation to independent rows.

## Reproduction artifacts

- Config: `configs/ne_typed_register_20m_shared_all.yaml`
- Implementation: `SharedResidualMicroCircuitBank` in `neural_engine/circuits.py`
- Run JSON/checkpoint: local under `results/runs/` and `results/checkpoints/`
