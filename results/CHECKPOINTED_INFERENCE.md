# Checkpoint-backed inference benchmark

This report measures the actual trained weights rather than freshly initialized
models. The checkpoint save/load path and analytical compute instrumentation
were added in commit `e729685`.

## Reproducibility

- Hardware: NVIDIA GeForce RTX 3060, 12 GB
- Runtime: `torch 2.6.0+cu124`, CUDA 12.4
- Seed: 17
- Training: 5,000 steps, batch size 128, balanced task sampling
- Inference: CUDA, batch size 128, 100 measured iterations, 3 warmup iterations
- Tests: `python -m pytest -q` -> `14 passed`

Training commands:

```powershell
python train.py --config configs/ne_20_v09_full.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v09_checkpoint_5000 --checkpoint results/checkpoints/ne20_v09_full.pt --log-every 1000
python train.py --config configs/ne_20_v12.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_checkpoint_5000 --checkpoint results/checkpoints/ne20_v12_full.pt --log-every 1000
python train.py --model baseline --steps 5000 --device cuda --balanced-train --run-id transformer_baseline_checkpoint_5000 --checkpoint results/checkpoints/transformer_baseline_5000.pt --log-every 1000
```

Inference command pattern:

```powershell
python benchmark.py --config <config.yaml> --checkpoint <checkpoint.pt> --device cuda --batch-size 128 --iterations 100 --balanced-batch
```

Checkpoint files are intentionally ignored by Git; the commands above make
them locally reproducible without adding large binary artifacts to the source
repository.

## Quality from the corresponding trained runs

| Model | Total params | Validation accuracy | Training seconds | Training peak VRAM |
|---|---:|---:|---:|---:|
| Dense Transformer | 20.582M | 51.02% | 556.8 | 1,396 MB |
| NE-V0.9 numeric, fixed depth | 20.246M | 71.59% | 164.6 | 516 MB |
| NE-V0.12 numeric + adaptive | 20.247M | **71.98%** | 165.3 | 516 MB |

NE-V0.9 and V0.12 both estimate 1.977M unique active parameters, or 9.77%
of their own stored capacity. V0.12's training validation schedule executed
1.597 of 3 recurrent steps on average; depth-1, depth-2, and depth-3 tasks
averaged 1.00, 2.00, and 2.99 steps respectively.

## Trained GPU inference

| Metric | Dense Transformer | NE-V0.9 fixed | NE-V0.12 adaptive |
|---|---:|---:|---:|
| Latency / batch | 34.031 ms | 8.929 ms | **8.322 ms** |
| Throughput | 3,761 samples/s | 14,336 samples/s | **15,382 samples/s** |
| Inference peak allocated VRAM | 157 MB | 162 MB | 162 MB |
| Average executed steps | dense | 3.000 | **1.563** |
| Active-step fraction | dense | 100.0% | **52.1%** |

On this RTX 3060 run, V0.12 is about 4.09x faster than the trained dense
baseline at batch size 128. This is an implementation result for the current
PyTorch code, not a hardware-independent guarantee; GPU kernel efficiency,
batch size, and memory layout can change the ratio.

## Analytical compute and weight-traffic proxy

The benchmark counts matrix-multiply MACs in the encoder, hierarchical router,
selected low-rank circuits, GRU update, and output head. It omits elementwise
operations, indexing, softmax, top-k, and kernel-launch overhead. Therefore
these are architectural estimates, not profiler measurements; one MAC is
roughly two floating-point operations under the common FLOP convention.

| Estimate per sample | Dense Transformer | NE-V0.9 fixed | NE-V0.12 adaptive |
|---|---:|---:|---:|
| Analytical MACs | 660.63M | 3.995M | **2.512M** |
| Unique active parameter bytes | 82.33 MB | 7.909 MB | 7.911 MB |
| Parameter-read proxy | 82.33 MB | 19.884 MB | **13.930 MB** |
| Parameter-read proxy vs dense | 100% | 24.2% | **16.9%** |

The unique-active number shows the capacity sparsity claim. The parameter-read
proxy is stricter: it includes the shared controller, router projections,
candidate keys, GRU/output weights, and selected circuit blocks for every
executed step. Adaptive execution lowers this proxy further, but it does not
pretend that routing and recurrent control are free.

## Decision

**V0.12 is the current recommended local model.** It preserves the best full
balanced accuracy measured so far, executes roughly half the recurrent routed
steps, and has the best trained GPU inference result in this benchmark. V0.9
remains useful as the fixed-depth quality reference.

The next engineering bottleneck is no longer “can the architecture activate a
small subset?” That signal is clear. The next step is a larger and stricter
generalization benchmark plus route-stability/overlap analysis, followed only
then by fused circuit kernels if the analytical advantage survives.
