# NE-V0.2 multi-address router

V0.2 adds a second independent hierarchical address head. The total candidate
pool remains 32 and the active circuit budget remains 8, so the change tests
whether multiple small addresses improve circuit coverage without increasing
selected computation.

## Controlled run

- Model: NE-20, 19.70M total parameters
- GPU: NVIDIA GeForce RTX 3060, 12 GB
- Runtime: `torch 2.6.0+cu124`, CUDA 12.4
- Seed: 17
- Training: 5,000 steps, batch size 128, balanced task sampling
- Command: `python train.py --config configs/ne_20_v02.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v02_balanced_cuda_5000 --log-every 1000`

## Result

| Metric | NE-V0.1 / NE-20 | NE-V0.2 / NE-20 | Dense Transformer |
|---|---:|---:|---:|
| Total parameters | 19.69M | 19.70M | 20.58M |
| Estimated active parameters | 1.422M | 1.434M | 20.58M |
| Active fraction | 7.22% | 7.28% | 100% |
| Balanced exact accuracy | 53.62% | **53.93%** | 51.02% |
| Validation loss | 1.5148 | **1.4646** | 1.6479 |
| Depth-1 accuracy | 80.90% | **82.16%** | 70.88% |
| Depth-2 accuracy | **17.45%** | 16.15% | 35.42% |
| Depth-3 accuracy | **7.94%** | 7.03% | 7.03% |
| Training time | **130.9 s** | 175.0 s | 543.3 s |
| Training throughput | **4,890.6/s** | 3,657.5/s | 1,178.1/s |
| Inference throughput | 20,195/s | 15,008/s | 3,721/s |

## Routing health

- Circuits used: `1394 / 1408`
- Dead circuit fraction: `0.99%`
- Routing entropy: `6.48` nats
- Maximum circuit load: `1.05%`
- Addresses: `2`
- Candidate pool: `32` total (`16` per address)
- Active circuits: `8`

## Decision

**Retain multi-address routing as an optional V0.2 variant.** It improves
coverage and slightly improves aggregate accuracy while preserving the central
active-computation property. It is not a free improvement: routing overhead
reduces throughput by about 26% relative to the one-address NE-20 model.

## NE-50 follow-up

The same two-address router was tested on the 49.77M-parameter NE-50 model:

| Metric | NE-50 one address | NE-50 two addresses |
|---|---:|---:|
| Balanced exact accuracy | **54.77%** | 53.10% |
| Estimated active parameters | 1.422M | 1.434M |
| Training throughput | **3,401/s** | 2,863/s |
| Circuits used | 2,755 / 3,712 | **3,318 / 3,712** |
| Dead circuit fraction | 25.78% | **10.61%** |

The extra address head improves coverage but lowers quality and throughput at
NE-50. Coverage alone is therefore not the objective; circuits must also be
specialized and composable for the selected state. The one-address router
remains the default for capacity scaling, while multi-address routing is kept
as an ablation for future load-balanced designs.
