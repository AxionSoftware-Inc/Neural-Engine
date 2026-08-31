# NE-20 parameter-matched comparison

This run checks whether the V0.1 result depends on storing 28M parameters. The
NE-20 model reduces the circuit bank to approximately 20M total parameters,
while keeping the same recurrent state, router, 8 active circuits, and 3
internal steps.

## Run

- GPU: NVIDIA GeForce RTX 3060, 12 GB
- Runtime: `torch 2.6.0+cu124`, CUDA 12.4
- Seed: 17
- Training: 5,000 steps, batch size 128, balanced task sampling
- NE command: `python train.py --config configs/ne_20.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_balanced_cuda_5000 --log-every 1000`
- Transformer command: `python train.py --model baseline --steps 5000 --device cuda --balanced-train --run-id baseline_balanced_cuda_5000 --log-every 1000`

## Result

| Metric | NE-20 | Dense Transformer |
|---|---:|---:|
| Total parameters | 19.69M | 20.58M |
| Estimated active parameters | 1.422M | 20.58M |
| Active fraction | 7.22% | 100% |
| Validation exact accuracy | **53.62%** | 51.02% |
| Validation loss | **1.5148** | 1.6479 |
| Depth-1 accuracy | **80.90%** | 70.88% |
| Depth-2 accuracy | **17.45%** | 35.42% |
| Depth-3 accuracy | **7.94%** | 7.03% |
| Training time | **130.9 s** | 543.3 s |
| Training throughput | **4,890.6 samples/s** | 1,178.1 samples/s |
| Peak VRAM | 502 MB | 1,396 MB |
| Inference throughput | **20,195 samples/s** | 3,721 samples/s |

The NE active set is approximately `6.9%` of the Transformer’s full active
parameter count while total stored capacity is within 4.3% of the baseline.

## Routing

- Circuits used: `1,339 / 1,408`
- Dead circuit fraction: `4.90%`
- Routing entropy: `6.28` nats
- Maximum circuit load: `1.12%`
- Router collapse: not observed

## Conclusion

**Positive signal confirmed at matched total capacity.** NE-20 retains the
quality/active-compute advantage and does not need the extra 28M total capacity
of the original V0.1 configuration. The main unresolved issue remains
multi-step composition, especially depth-2 tasks where the dense baseline is
stronger.

Next experiment: NE-50 and NE-100 with the same approximately fixed active
budget, to test whether dormant capacity improves quality without proportional
active compute growth.
