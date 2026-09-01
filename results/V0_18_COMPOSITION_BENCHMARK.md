# V0.18 composition benchmark

V0.18 adds an explicit arithmetic program benchmark for the architectural
hypothesis. Each example contains two operation tokens and three numeric
operands. The target is the result of the ordered composition, modulo 64:

```text
(a op1 b) op2 c
```

The training split contains seven of the nine ordered operation pairs. The
held-out split hides `add -> multiply` and `multiply -> add`, so those pairs
test composition rather than memorization of a task ID. A separate
learnability-control configuration exposes all nine pairs during training and
reports a fresh balanced sample from the same all-pairs distribution.

## Reproduction

All runs below used an RTX 3060 12 GB, CUDA, seed 17, batch size 128, and 5,000
steps. Checkpoints are local and ignored by Git.

```powershell
python train_composition.py --config configs/ne_composition_v0.yaml --steps 5000 --device cuda --run-id ne_composition_v0_checkpoint_5000 --output results/runs --examples-per-task 64 --log-every 1000 --checkpoint results/checkpoints/ne_composition_v0_full.pt
python train_composition.py --config configs/ne_composition_stage.yaml --steps 5000 --device cuda --run-id ne_composition_stage_checkpoint_5000 --output results/runs --examples-per-task 64 --log-every 1000 --checkpoint results/checkpoints/ne_composition_stage_full.pt
python train_composition.py --config configs/transformer_composition.yaml --steps 5000 --device cuda --run-id transformer_composition_checkpoint_5000 --output results/runs --examples-per-task 64 --log-every 1000 --checkpoint results/checkpoints/transformer_composition_full.pt
python train_composition.py --config configs/ne_composition_all.yaml --steps 5000 --device cuda --run-id ne_composition_all_5000 --output results/runs --examples-per-task 64 --log-every 1000 --checkpoint results/checkpoints/ne_composition_all_5000.pt
python train_composition.py --config configs/transformer_composition_all.yaml --steps 5000 --device cuda --run-id transformer_composition_all_5000 --output results/runs --examples-per-task 64 --log-every 1000 --checkpoint results/checkpoints/transformer_composition_all_5000.pt
```

## Results

The reported active estimate includes shared computation, routing keys for
the local candidate pool, and eight selected low-rank circuits per internal
step. The dense baseline touches its full parameter set.

| Run | Total params | Active params | Active fraction | Train accuracy | Evaluation accuracy | Evaluation split | Train samples/s |
|---|---:|---:|---:|---:|---:|---|---:|
| NE, 7 seen pairs | 20.367M | 2.098M | 10.30% | 20.09% | 12.50% | 2 held-out pairs | 3,658 |
| NE + stage loss 0.1 | 20.367M | 2.098M | 10.30% | 20.09% | 14.84% | 2 held-out pairs | 3,392 |
| Dense Transformer, 7 seen pairs | 20.573M | 20.573M | 100% | 10.27% | 9.38% | 2 held-out pairs | 2,913 |
| NE, all 9 pairs control | 20.367M | 2.098M | 10.30% | 20.83% | 17.88% | all pairs | 3,779 |
| Dense Transformer, all 9 pairs control | 20.573M | 20.573M | 100% | 11.81% | 10.24% | all pairs | 3,039 |

The all-pairs control is intentionally not a generalization result. It asks
whether the task can be learned at all under the current 5,000-step recipe.
The random exact-match baseline is 1.56% for 64 classes. Both models learn
well above random and reduce loss, but both remain far from a solved
composition task. NE is ahead of the dense baseline in this screening (+9.02
points on the all-pairs training sample), while touching roughly one tenth of
the estimated active parameters. This is a useful efficiency signal, not yet
a claim of final architectural superiority.

Stage supervision raises the held-out score from 12.50% to 14.84% in this
single seed, while leaving the measured training accuracy unchanged. It is
therefore kept as an optional composition recipe rather than enabled by
default.

## Decision

The experiment did not invalidate Neural Engine. It showed two things:

1. The main numeric benchmark still has a strong positive NE signal, but the
   new arithmetic composition benchmark is much harder than the current
   5,000-step training budget can solve.
2. The comparison is not yet a clean generalization win because the models
   underfit even when all operation pairs are visible.

Before a serious 300–500M training run, use a short learnability ladder:

```text
add/subtract-only depth 2  ->  add/subtract-only depth 3
                         ->  full arithmetic depth 3
                         ->  held-out operation pairs
```

Run the ladder for 10k–20k steps with two seeds. Move to a large model only
when the first stages approach a high training ceiling and the held-out gap is
measurable.

## 300–500M feasibility plan

With the current rank-16 circuit shape, 22,800 circuits put the model near
300M total parameters. Keeping eight active circuits and the same
shared controller gives an estimated active path near 2.1M parameters, or
about 0.7% of total capacity. Approximately 39,000 circuits would approach
500M and reduce that fraction below 0.5%.

The 300M screen is configured in `configs/ne_300m_screen.yaml`. It uses a
depth-5, branch-8 router because `8^5 = 32,768` address leaves cover the
22,800-circuit bank. A 500M version needs a depth-6 router or another coverage
strategy; the current exact soft coverage distribution should be replaced by a
factorized or hashed regularizer before enabling it at that size.

Do not begin with a long 500M AdamW run on the RTX 3060. First measure a real
300M forward/backward step. If memory is acceptable, use a small batch with
gradient accumulation and an 8-bit optimizer or Adafactor for training. Full
FP16 inference is expected to be much easier than full-parameter training.

The first CUDA screens are now complete:

| Screen | Total params | Train batch | Peak allocated VRAM | One-step throughput | Average active fraction |
|---|---:|---:|---:|---:|---:|
| NE-300M | 299.544M | 16 | 5,735 MiB | 26.76 samples/s | 0.701% |
| NE-500M | 505.832M | 8 | 9,665 MiB | 11.49 samples/s | 0.421% |

Both screens completed a real forward/backward/AdamW update on the RTX 3060.
The 500M result leaves substantially less VRAM headroom and is much slower,
but it is technically runnable with the reduced batch. These are feasibility
measurements only; they do not establish that the larger models learn better.
