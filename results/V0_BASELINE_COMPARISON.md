# V0 local smoke comparison

This is a code-correctness and learning-signal smoke test, not the final V0
research result. The run was intentionally short and used the CPU fallback
because the installed PyTorch wheel was CPU-only.

## Reproducibility

- Commit SHA: `f2be7c1`
- GPU detected by host: NVIDIA GeForce RTX 3060, 12288 MiB
- Runtime used: `cpu` (`torch 2.13.0+cpu`, CUDA unavailable in Python)
- Install: `python -m pip install -r requirements.txt`
- Tests: `python -m pytest -q`
- Neural Engine: `python train.py --model ne --smoke --steps 20 --device cpu --run-id ne_smoke_cpu --log-every 5`
- Baseline: `python train.py --model baseline --smoke --steps 20 --device cpu --run-id baseline_smoke_cpu --log-every 5`
- Inference: `python benchmark.py --model ne --smoke --device cpu --batch-size 32 --iterations 5`

## Results

| Metric | Neural Engine V0 | Dense baseline |
|---|---:|---:|
| Smoke parameters | 459,352 | 293,952 |
| Full-config parameters | 28,021,344 | 20,582,208 |
| Estimated active parameters (full config) | 1,396,320 | 20,582,208 |
| Estimated active fraction (full config) | 4.98% | 100% |
| Training steps | 20 | 20 |
| Training loss, first → last | 4.4535 → 3.8739 | 4.2502 → 3.9176 |
| Validation loss | 3.9803 | 3.9152 |
| Validation exact accuracy | 9.77% | 11.33% |
| Training samples/sec | 5,497.5 | 2,918.1 |
| Peak VRAM | Not measured (CPU) | Not measured (CPU) |

NE-V0 routing used a 3-level tree in smoke mode, 16 local candidates, 4 active
circuits, and 2 internal recurrent steps. The full configuration uses 4 levels,
32 local candidates, 8 active circuits, and 3 steps.

## Tests and interpretation

- `3 passed` unit tests: generator shape/label validity, structured local routing,
  and forward/backward gradient flow into both circuits and router parameters.
- Both models produced a decreasing training loss over 20 steps.
- Exact accuracy is still near the short-run baseline and therefore does not
  establish useful task competence.
- Classification quality is not yet a positive architectural signal; the next
  run should use a CUDA-enabled PyTorch install and longer, fixed-seed training.
- Wall-clock speed is not comparable to the intended GPU experiment. The active
  parameter estimate is algorithmic instrumentation only, not a claim of faster
  PyTorch kernels.

Conclusion: **WEAK SIGNAL / HARNESS VALIDATED**, not a model-quality result.
