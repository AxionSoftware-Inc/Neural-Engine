# V0.186 Qwen conditional-width oracle

## Question

Can the quality/capacity problem be solved by spending more active parameters
only on difficult tokens? This is an oracle screen for the idea, before
training a deployable difficulty predictor.

## Method

Layer 26 of frozen Qwen3-0.6B was replaced by three independently distilled,
attention-free Qwen-like SwiGLU children with intermediate widths 192, 768,
and 1536. These correspond to 6.25%, 25%, and 50% of the teacher's 3072-wide
FFN. The children were trained for 300 steps on four calibration batches from
`data/qwen_calibration.txt`.

For each held-out token, an oracle computed each child's local reconstruction
error against the frozen teacher output. It selected the width minimizing

```text
normalized_local_error + lambda * active_width_fraction
```

The selected child alone was executed by the hard wrapper. This route uses
teacher outputs and is therefore an optimistic upper bound, not a deployable
router. Evaluation uses the independent `data/qwen_eval.txt` corpus, float32
CUDA, 2 batches, batch size 8, sequence length 128. The target gate is CE
delta `<= +0.05` at no more than 50% average active width.

## Results

| lambda | Average active width | Route 192 / 768 / 1536 | Held-out CE delta | Top-1 agreement | Gate |
|---:|---:|---:|---:|---:|:---:|
| 0.00 | 34.03% | 18.75% / 31.05% / 50.20% | +0.0636 | 89.11% | FAIL |
| 0.05 | 29.78% | 25.34% / 36.52% / 38.13% | +0.0659 | 88.92% | FAIL |
| 0.10 | 25.49% | 34.42% / 37.79% / 27.78% | +0.0679 | 88.67% | FAIL |
| 0.25 | 17.25% | 56.40% / 32.32% / 11.28% | +0.0741 | 88.04% | FAIL |
| 0.50 | 11.82% | 76.86% / 18.21% / 4.93% | +0.0809 | 87.70% | FAIL |
| 1.00 | 8.83% | 90.14% / 6.93% / 2.93% | +0.0873 | 87.40% | FAIL |
| 2.00 | 7.61% | 95.17% / 3.03% / 1.81% | +0.0900 | 87.16% | FAIL |
| 5.00 | 7.01% | 97.71% / 0.98% / 1.32% | +0.0877 | 87.11% | FAIL |

The child bank stores 81.25% as many scalars as the parent before any router
metadata. The best oracle point improves on a single fixed-width compact child
at similar average compute, but it still misses the quality gate even though
the route is allowed to see the teacher's true local error. The route changes
width with the penalty, so the mechanism is not dead; the available compact
functions are simply not accurate enough for this layer/corpus.

## Decision

**REJECTED as the immediate quality solution.** Do not train a difficulty
predictor on top of these three children and do not scale this exact adaptive
width recipe to larger Qwen models. Since even a teacher-informed oracle fails
at `<=50%` active width, a learned router could not be expected to repair the
representation gap.

The main path returns to raw Qwen neuron-preserving groups with signed/output
mixing, where the exact-subset oracle has already shown a positive 2--4-layer
signal. The open problem there is route/cascade generalization and depth, not
whether a smaller generic SwiGLU can substitute for the teacher FFN.

## Reproduction

```powershell
python benchmark_qwen_adaptive_width_oracle.py `
  --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 `
  --layer-index 26 --widths 192,768,1536 `
  --lambdas 0 0.05 0.1 0.25 0.5 1 2 5 `
  --batch-size 8 --sequence-length 128 --train-batches 4 --eval-batches 2 `
  --calibration-text-file data/qwen_calibration.txt `
  --eval-text-file data/qwen_eval.txt --steps 300 --learning-rate 3e-3 `
  --seed 2026 `
  --output results/runs/qwen_oracle_adaptive_width_layer26_diverse_seed2026.json
```

## Artifacts

- `benchmark_qwen_adaptive_width_oracle.py`
- `tests/test_adaptive_width_oracle.py`
- `results/runs/qwen_oracle_adaptive_width_layer26_diverse_seed2026.json`
