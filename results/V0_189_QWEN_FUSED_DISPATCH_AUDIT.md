# V0.189 Qwen fused-dispatch kernel audit

## Question

Can a custom CUDA kernel remove the Python/expert-dispatch overhead from the
selected-token Qwen bank without changing the selected circuit computation?

## Protocol

The existing V0.188 setup was kept: Qwen3-0.6B, float32 CUDA, contiguous
E=8/K=6 groups, rank-64 cross-group correction, and the same selected-token
contract. The machine has an RTX 3060 (compute capability 8.6), CUDA toolkit
13.3, PyTorch 2.6.0+cu124, and Visual Studio 2022 C++ Build Tools.

Two opt-in controls were implemented:

- `packed-fused`: one gate+value GEMM per selected expert, followed by the
  existing output GEMM;
- `fused`: a custom CUDA extension with one block per selected token/group
  pair. It computes the SwiGLU coefficient in shared memory and atomically
  accumulates the weighted output.

The normal `grouped` default is unchanged. The custom extension is lazy-built
only when `--dispatch-mode fused` is requested.

## Results

### PyTorch packed-fused control

One replaced layer, batch 8 × sequence 128, 10 timing iterations:

| Backend | Parent ms/batch | Sparse ms/batch | Sparse / parent |
|---|---:|---:|---:|
| packed | 236.89 | 269.78 | 1.139x |
| packed-fused | 236.82 | 269.71 | 1.139x |

The 8-layer, 5-iteration smoke gave `509.91 ms` for packed-fused versus the
V0.188 packed reference `505.88 ms` at 20 iterations. This is noise-level to
slightly worse, not a speedup. The short 8-layer run also had alpha=0 CE
delta `+0.0864`; this quality number is not used as a new quality claim
because its evaluator batch count differed from the quality controls.

### Custom CUDA kernel

The extension compiled successfully after importing the VS x64 environment and
adding `/Zc:preprocessor` for CUDA 13.3. A direct kernel-only test with the
realistic shape `N=1024, H=1024, E=8, K=6, group=384` took `832.6 ms` for one
selected dispatch. The small random parity smoke matched the PyTorch reference
within `5.4e-5` absolute output error and passed `allclose(atol=3e-5,
rtol=3e-5)`, but the kernel is far slower than cuBLAS-backed `F.linear`.

The first end-to-end Qwen attempt did not reach a valid timing result after
more than 90 seconds and was interrupted. It is not counted as a benchmark
number; the direct kernel result is sufficient to reject this implementation.

## Decision

**Rejected for adoption.** Combining the first two projections does not
recover runtime. The naive one-block-per-pair kernel preserves the selected
path but destroys GEMM tiling and is approximately an order of magnitude too
slow on the target shape. `grouped` remains the default, while `packed`,
`packed-fused`, and `fused` remain opt-in diagnostic backends.

This narrows P-006: the remaining speed problem is not just Python dispatch;
it requires a tiled/grouped GEMM implementation (for example CUTLASS,
cuBLAS grouped GEMM, or a Linux/Triton-capable environment). No quality claim
or model-scale decision follows from V0.189.

## Reproduction

```powershell
python -u benchmark_qwen_multi_layer_transplant.py `
  --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 `
  --layers 19,20,21,22,23,24,25,26 --child-kind qwen-transfer-sparse `
  --num-experts 8 --active-experts 6 --calibration-rank 64 `
  --calibration-mode cross-group --partition-mode contiguous `
  --route-source router --hard-route-scale 6 --steps 0 `
  --train-batches 1 --eval-batches 4 --batch-size 8 --sequence-length 128 `
  --alphas 1 0 --calibration-text-file data/qwen_calibration.txt `
  --eval-text-file data/qwen_eval.txt --timing-warmup 2 `
  --timing-iterations 5 --dispatch-mode packed-fused --seed 2026 `
  --output results/runs/qwen_dispatch_8layers_k6_packed_fused_seed2026.json
```

The custom extension sources are `neural_engine/qwen_fused_dispatch.cpp` and
`neural_engine/qwen_fused_dispatch.cu`.
