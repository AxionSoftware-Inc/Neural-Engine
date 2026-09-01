# V0.19 fair numeric Transformer baseline

`taklif.md` identifies baseline fairness as a priority: the original dense
Transformer used a plain token embedding while Neural Engine used the
structured 64-modular numeric/Fourier value encoder. V0.19 moves that encoder
into a shared helper and enables the exact same value representation in the
Transformer control.

The Transformer still uses dense self-attention. Only the value frontend is
shared; the recurrent state, hierarchical router, and micro-circuit bank
remain NE-specific.

## Reproduction

The main benchmark uses the existing task-aware operand-combination split. All
runs used an RTX 3060 12 GB, CUDA, seed 17, balanced batches, batch size 128,
and 5,000 steps.

```powershell
python train.py --config configs/ne_20_v12_combo_heldout.yaml --steps 5000 --device cuda --balanced-train --run-id ne20_v12_combo_heldout_cuda_5000 --output results/runs --log-every 1000
python train.py --config configs/transformer_combo_heldout.yaml --steps 5000 --device cuda --balanced-train --run-id transformer_combo_heldout_cuda_5000 --output results/runs --log-every 1000
python train.py --config configs/transformer_20_v12_numeric.yaml --steps 5000 --device cuda --balanced-train --run-id transformer_20_numeric_fair_5000 --output results/runs --log-every 1000 --checkpoint results/checkpoints/transformer_20_numeric_fair_5000.pt
```

The composition control was also rerun with the shared numeric frontend:

```powershell
python train_composition.py --config configs/transformer_composition_fair.yaml --steps 5000 --device cuda --run-id transformer_composition_fair_5000 --output results/runs --examples-per-task 64 --log-every 1000 --checkpoint results/checkpoints/transformer_composition_fair_5000.pt
```

## Main benchmark results

| Model | Numeric frontend | Total params | Active fraction | Train split accuracy | Held-out accuracy | Training samples/s |
|---|---|---:|---:|---:|---:|---:|
| NE-V0.12 | structured Fourier | 20.247M | 10.07% average | 71.41% | 55.68% | 3,666 |
| Dense Transformer, original | plain embedding | 20.582M | 100% | 51.02% | 50.81% | 1,116 |
| Dense Transformer, fair | structured Fourier | 20.545M | 100% | **72.37%** | **65.91%** | 1,113 |

The fair Transformer improves by 21.35 points on the train split and 15.10
points on the held-out split relative to the old plain-embedding control. This
shows that the earlier NE-versus-Transformer quality gap was confounded by
input representation.

On this protocol, the fair Transformer is 0.96 points ahead of NE on the
train split and 10.23 points ahead on held-out combinations. NE still trains
at roughly 3.3x the sample throughput and estimates only 10.07% average active
parameters, so the strongest remaining claim is a compute/quality Pareto
question rather than unconditional quality superiority.

Checkpoint inference at batch size 128 gives the following controlled speed
measurement:

| Model | Samples/s | Latency/batch | Estimated parameter-read fraction |
|---|---:|---:|---:|
| NE-V0.12 | **15,136** | **8.46 ms** | 17.20% |
| Fair numeric Transformer | 3,755 | 34.09 ms | 100% |

Thus the fair input correction removes the old quality confound without
removing NE's measured inference efficiency advantage. The speed result is
still a PyTorch implementation measurement, not a custom sparse-kernel claim.

## Composition control

The new two-operation benchmark is harder and both models underfit at 5,000
steps:

| Model | Numeric frontend | Seen-pair accuracy | Held-out-pair accuracy |
|---|---|---:|---:|
| NE-V0.12 | structured Fourier | 20.09% | 12.50% |
| Dense Transformer, fair | structured Fourier | 12.28% | 4.69% |

This supports NE as the stronger short-screen learner on that particular
composition setup, but the absolute scores are too low for a final
generalization claim. The composition ladder must be trained longer and with
two seeds before it gates model scaling.

## Decision

The fair baseline changes the research framing:

1. Do not use the old plain-embedding Transformer as the main quality control.
2. Keep NE because it preserves the active-compute and throughput advantage,
   and because it remains stronger on the early composition screen.
3. Do not claim that NE is more accurate than a Transformer until the fair
   control is included.
4. Before a long 300M/500M quality run, validate sparse/lazy AdamW against
   current AdamW on the 20M model and finish the composition ladder.
