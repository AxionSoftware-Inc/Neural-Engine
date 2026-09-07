# V0.185 Qwen teacher-derived functional basis

## Question

Can a Qwen FFN be compressed into a smaller attention-free SwiGLU whose
coordinates are derived from the teacher's gate/value activation covariance?
This tests the representation part of `taklif15.md`; it does not add a new
router or copy contiguous Qwen neuron slices.

## Method

Qwen3-0.6B was frozen and only late layer 26 was replaced. The parent is a
bias-free Qwen SwiGLU with hidden size 1024 and intermediate size 3072
(9,437,184 scalar parameters). The child keeps the same gated algebra but
uses a reduced intermediate width `r`.

For the teacher-derived initialization, gate and value activations from
`data/qwen_calibration.txt` were standardized, their joint neuron-space
covariance was eigendecomposed, and the top `r` orthogonal directions were
used to project gate, value, and down weights. The child was then distilled
from parent hidden-state/output pairs. The random rows below are a same-width
control, not a new architecture.

The stringent protocol uses float32 CUDA, the independent
`data/qwen_eval.txt` held-out corpus, 4 calibration batches, 2 evaluation
batches, batch size 8, sequence length 128, and the local CE gate of `+0.05`.
The 192/384 runs use 300 steps; 768/1536 controls use 500 steps.

## Results

| Initialization | r | Active/parameter fraction | Steps | Held-out CE delta | Teacher top-1 | Local MSE | Gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| teacher activation basis | 192 | 6.25% | 300 | +0.1123 | 87.01% | 4.8147 | FAIL |
| random control | 192 | 6.25% | 300 | +0.1067 | 87.11% | 4.2255 | FAIL |
| teacher activation basis | 384 | 12.5% | 300 | +0.1333 | 86.28% | 5.4503 | FAIL |
| random control | 384 | 12.5% | 300 | +0.0972 | 88.13% | 4.0683 | FAIL |
| random control | 768 | 25% | 500 | +0.0790 | 88.53% | 4.2800 | FAIL |
| random control | 1536 | 50% | 500 | +0.0832 | 88.13% | 3.7764 | FAIL |

The same teacher-basis r=192 command on the historical narrow demo text gave
`-0.0060` CE delta and 98.24% top-1 agreement. Its r=192 seed-2027 rerun was
identical because the teacher-derived projection and training path are fully
deterministic; it is not counted as an independent seed confirmation. The
independent held-out corpus reverses that apparent positive result.

## Decision

**REJECTED as a quality solution.** Activation-covariance projection is not a
useful Qwen functional basis at these budgets: it is slightly worse than the
same-width random child at both tested widths, and increasing compact width
from 12.5% to 50% does not reach the `+0.05` gate in this protocol. This is
not evidence that all teacher-derived bases are impossible, but this specific
Galerkin projection is closed and should not be scaled to larger Qwen models.

The result also separates two bottlenecks. A compact full child is not
equivalent to the validated 50%-active copied-group plus cross-group path in
V0.161: reducing the nonlinear feature basis loses more function than the
grouped path on this held-out cascade. The next experiment must therefore
either preserve teacher neuron nonlinearities while improving routing, or
explicitly test conditional width so hard tokens can spend more active
compute without forcing every token onto the largest child.

## Reproduction

```powershell
python benchmark_qwen_teacher_basis_swiglu.py `
  --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 `
  --layer-index 26 --inner-size 192 --initialization teacher-basis `
  --batch-size 8 --sequence-length 128 --train-batches 4 --eval-batches 2 `
  --calibration-text-file data/qwen_calibration.txt `
  --eval-text-file data/qwen_eval.txt --steps 300 --learning-rate 3e-3 `
  --seed 2026 `
  --output results/runs/qwen_teacher_basis_swiglu_layer26_r192_diverse_seed2026.json
```

## Artifacts

- `benchmark_qwen_teacher_basis_swiglu.py`
- `tests/test_teacher_basis_swiglu.py`
- `results/runs/qwen_teacher_basis_swiglu_layer26_r192_diverse_seed2026.json`
- `results/runs/qwen_teacher_basis_swiglu_layer26_r384_diverse_seed2026.json`
- `results/runs/qwen_teacher_basis_swiglu_layer26_r192_random_diverse_seed2026.json`
- `results/runs/qwen_teacher_basis_swiglu_layer26_r384_random_diverse_seed2026.json`
- `results/runs/qwen_teacher_basis_swiglu_layer26_r768_random_diverse_seed2026_steps500.json`
- `results/runs/qwen_teacher_basis_swiglu_layer26_r1536_random_diverse_seed2026_steps500.json`
