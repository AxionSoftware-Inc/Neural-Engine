# V0.20 lazy AdamW prototype

`taklif.md` notes that the current `AdamW(model.parameters())` trainer is
not sparse: the forward path selects a small circuit set, but the optimizer
still owns and updates the full parameter bank. V0.20 adds a correctness-first
`LazyAdamW` prototype.

The prototype keeps ordinary AdamW updates for dense controller parameters. For
the circuit bank and router key table it:

- receives the selected route IDs from the model;
- gathers only those rows and their gradients;
- keeps first/second moments only for rows touched so far;
- applies weight decay only to touched rows;
- updates the selected rows with batched indexed tensor operations.

The last point matters: the initial row-by-row implementation matched quality
but took 122.5 seconds for 500 steps. The batched implementation reduced the
same probe to 18.9 seconds.

## Reproduction

All runs used an RTX 3060 12 GB, CUDA, seed 17, task-balanced batches, batch
size 128, and the same 20M V0.12 held-out value-combination protocol.

```powershell
python train.py --config configs/ne_20_v12_combo_heldout.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_dense_probe_5000 --output results/runs --log-every 1000
python train.py --config configs/ne_20_v12_combo_heldout_lazy.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_lazy_batched_5000 --output results/runs --log-every 1000 --checkpoint results/checkpoints/ne20_v12_lazy_batched_5000.pt
python train.py --config configs/ne_100_v12_coverage_lazy.yaml --steps 5000 --device cuda --balanced-train --run-id ne100_lazy_5000 --output results/runs --log-every 1000 --checkpoint results/checkpoints/ne100_lazy_5000.pt
python train.py --config configs/ne_300m_screen.yaml --steps 200 --device cuda --balanced-train --run-id ne300m_dense_screen_200 --output results/runs --log-every 50
python train.py --config configs/ne_300m_screen_lazy.yaml --steps 200 --device cuda --balanced-train --run-id ne300m_lazy_screen_200 --output results/runs --log-every 50
```

The dense reference row below is the existing `ne20_v12_combo_heldout_cuda_5000`
run; the lazy run is saved in
`results/runs/ne20_v12_lazy_batched_5000.json`.

## Results

| Optimizer | Total params | Avg active fraction | Train accuracy | Held-out accuracy | Training time | Training samples/s | Lazy state rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense AdamW | 20.247M | 10.07% | 71.41% | **55.68%** | **174.6 s** | **3,666** | N/A |
| Batched LazyAdamW | 20.247M | 10.07% | **72.06%** | 55.36% | 183.1 s | 3,495 | 5,632 |

The lazy run touches 3,484 routed rows on its final update and has 5,632
state rows across the four lazy parameter tensors (circuit down/up/bias and
router keys). Quality is within 0.32 percentage points on held-out data and
the training slowdown is about 4.7% in this small controlled run.

The same A/B was then repeated at 100M with the coverage-aware NE-100
configuration:

| Optimizer | Total params | Avg active fraction | Validation accuracy | Training time | Training samples/s | Dead circuit fraction |
|---|---:|---:|---:|---:|---:|---:|
| Dense AdamW | 100.466M | 2.03% | **72.14%** | 357.0 s | 1,793 | 6.22% |
| Batched LazyAdamW | 100.466M | 2.03% | 71.67% | 306.8 s | **2,086** | 6.12% |

The 100M quality difference is 0.47 points in one seed and the routing
statistics remain effectively unchanged. The apparent timing improvement is
not treated as a claim because these are separate wall-clock runs; the
important result is that lazy updates remain numerically viable at 100M.
The lazy optimizer state contains 30,208 row entries at the end of this run,
which is four tensors times the 7,552-circuit bank: this workload eventually
uses nearly the whole bank, so it is not yet a memory-saving demonstration.

Finally, the same optimizer path was screened on the 300M configuration for
200 steps. This is an early underfit screen, not a quality comparison:

| Optimizer | Total params | Train batch | Validation accuracy | Final loss | Training time | Training samples/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense AdamW | 299.544M | 16 | **19.38%** | 4.4237 | 28.9 s | 111 | 5,735 MiB |
| Batched LazyAdamW | 299.544M | 16 | 19.30% | 4.4304 | **14.7 s** | **218** | **5,074 MiB** |

The two models start from the same initialization and see the same minibatch
stream. LazyAdamW is about 2x faster and uses about 660 MiB less peak VRAM in
this short test, with indistinguishable early quality. The low absolute score
is expected at 200 steps and must not be used as a 300M scaling result.

## Limitations

This is not yet the final 8B sparse trainer:

- all model weights still reside on the GPU;
- no CPU-RAM circuit offload or GPU cache exists;
- no asynchronous prefetch or H2D accounting exists;
- after enough training, all 1,408 circuit rows can become resident in the
  lazy state, so the 20M run does not demonstrate a large memory reduction;
- touched-row-only weight decay is intentionally a lazy approximation to
  dense AdamW and needs a longer numerical validation.

The key result is narrower: route-indexed, batched lazy updates can preserve
the current quality trajectory without the severe Python-loop slowdown of the
first prototype.

## Decision

Keep `optimizer: lazy_adamw` as an experimental option. Do not make it the
default yet. The next validation is a 300M fixed-active screening run,
followed by a CPU-RAM/offload prototype that reports cache hit rate, H2D
bytes, GPU idle fraction, and optimizer state bytes touched. Only then should
a 500M quality run use the sparse trainer as its primary training path.
