# V0 CUDA controlled comparison

This is the first meaningful local result for the attention-free Neural Engine
architecture. Both models used the same synthetic task representation, seed,
optimizer, batch size, 5,000 steps, and balanced task sampling.

## Reproducibility

- Code commit: `bae8a82`
- Host GPU: NVIDIA GeForce RTX 3060, 12 GB
- Runtime: `torch 2.6.0+cu124`, CUDA 12.4
- Install: `python -m pip install -r requirements-cuda.txt`
- Tests: `python -m pytest -q` → `3 passed`
- Neural Engine: `python train.py --model ne --steps 5000 --device cuda --balanced-train --run-id ne_v0_balanced_cuda_5000 --log-every 1000`
- Transformer: `python train.py --model baseline --steps 5000 --device cuda --balanced-train --run-id baseline_balanced_cuda_5000 --log-every 1000`
- Inference benchmark: `python benchmark.py --model {ne|baseline} --device cuda --batch-size 128 --iterations 20`

## Main result

| Metric | Neural Engine V0 | Dense Transformer |
|---|---:|---:|
| Total parameters | 28,021,344 | 20,582,208 |
| Estimated active parameters | 1,396,320 | 20,582,208 |
| Active fraction of own model | 4.98% | 100% |
| NE active / Transformer active | 6.78% | 100% |
| Active circuit blocks | 8 per internal step | N/A |
| Internal steps | 3 | N/A |
| Validation exact accuracy | **52.71%** | 51.02% |
| Validation loss | **1.5193** | 1.6479 |
| Depth-1 accuracy | **81.60%** | 70.88% |
| Depth-2 accuracy | 10.29% | **35.42%** |
| Depth-3 accuracy | **8.46%** | 7.03% |
| Training time | **141.3 s** | 543.3 s |
| Training throughput | **4,528.6 samples/s** | 1,178.1 samples/s |
| Peak VRAM | 614 MB | 1,396 MB |
| Inference throughput | **19,026 samples/s** | 3,977 samples/s |

The total parameter counts are in the same 20–30M scale, but are not yet
exactly matched: NE-V0 has 1.36x more stored capacity. The important result is
that its estimated active set is 14.7x smaller than the dense baseline while
reaching slightly higher balanced accuracy in this run.

## Task-level accuracy

| Task | NE-V0 | Transformer |
|---|---:|---:|
| add | 96.88% | 10.16% |
| subtract | 49.22% | 6.64% |
| multiply | 89.84% | 22.27% |
| greater_than | 53.52% | 100.00% |
| less_equal | 48.02% | 99.22% |
| xor_parity | 100.00% | 100.00% |
| max3 | 99.22% | 99.61% |
| median3 | 98.05% | 100.00% |
| min3 | 99.61% | 100.00% |
| reverse_sum | 1.95% | 3.91% |
| lookup | 25.39% | 100.00% |
| chain3 | 3.52% | 2.34% |
| compose_add_mul | 15.63% | 10.55% |
| compose_if | 4.69% | 7.81% |
| state_machine | 5.08% | 2.73% |

## Routing health

- Circuits used in validation: `1904 / 2048`
- Dead circuit fraction: `7.03%`
- Routing entropy: `6.82` nats
- Maximum circuit load: `0.75%`
- Router: 4-level branching tree, branch factor 8, local candidate pool 32
- Selected blocks: 8 per internal step

The router is not collapsed onto a tiny fixed subset. Most of the circuit bank
participates, while each individual example still activates only a small local
set.

## Interpretation

Conclusion: **STRONG ARCHITECTURAL SIGNAL, INCOMPLETE TASK RESULT**.

The central hypothesis has passed its first falsifiable test: comparable useful
performance was obtained with a much smaller estimated active parameter set,
without attention or Transformer blocks. It is not yet a final victory because
both models are weak on multi-hop/composed tasks, total capacity is not exactly
matched, and active parameters are currently an analytical estimate rather than
a custom-kernel hardware measurement.

## Next experiment

The next version should keep the routing idea and improve composition:

1. add explicit multi-step curriculum and harder held-out compositions;
2. compare NE-20 and a parameter-matched dense baseline;
3. log active weight bytes/FLOPs and route stability;
4. test 20, 50, and 100M total capacity at approximately fixed active budget;
5. only then consider fused CUDA kernels.
