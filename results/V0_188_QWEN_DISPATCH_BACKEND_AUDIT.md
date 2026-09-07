# V0.188 Qwen selected-group dispatch backend audit

## Question

The strongest Qwen sparse path has a quality signal, but its Python grouped
execution is slower than the dense parent. Can another available PyTorch
dispatch layout recover the saved compute, and can `torch.compile` fuse the
path in the current Windows environment?

## Protocol

This is a runtime-only smoke: no child or router training (`steps=0`). The
same frozen Qwen3-0.6B layers 19--26, E=8/K=6 raw-neuron bank, rank-64
cross-group wrapper, batch size 8, sequence length 128, four held-out batches,
float32 CUDA and 20 timing iterations are used for each backend. The router
and selected route are held fixed by the same zero-initialized smoke setup.
Compilation warm-up is excluded from reported timing.

## Results

| Backend | Parent ms/batch | Sparse ms/batch | Sparse / parent | Status |
|---|---:|---:|---:|---|
| grouped | 237.27 | 506.60 | 2.135x | valid |
| token-loop | 237.50 | 504.61 | 2.125x | valid |
| packed, memory-safe per-group batched linear | 237.09 | 505.88 | 2.134x | valid |

The first packed implementation was rejected before timing because gathering a
full output weight tensor for every token/group pair requested approximately
9 GB of additional memory. It was corrected to process each group with
batched `F.linear`; the corrected implementation matches token-loop output to
`1e-6` and does not OOM, but it is not faster.

An opt-in `torch.compile(mode="reduce-overhead")` attempt failed before timing
because the installed Windows PyTorch build has no usable Triton installation:
`Cannot find a working triton installation`. No compiled speed claim is made.

## Decision

**No runtime win in the current environment.** Grouped, token-loop, and the
memory-safe packed fallback all remain about 2.13x slower than the dense parent
on this small RTX 3060 test. The Qwen selected-group quality path is therefore
not yet a deployment speedup. The implementation is kept as an opt-in,
numerically checked fallback; a real speed gate requires a fused CUDA/Triton
kernel on a supported environment or a custom extension.

This does not change the quality decisions in V0.174/V0.187 and does not
justify scaling to 700M/1B. It narrows P-006: the remaining runtime problem is
backend fusion and dispatch overhead, not simply choosing between the current
three Python layouts.

## Reproduction

```powershell
python -u benchmark_qwen_multi_layer_transplant.py `
  --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 `
  --layers 19,20,21,22,23,24,25,26 --child-kind qwen-transfer-sparse `
  --num-experts 8 --active-experts 6 --calibration-rank 64 `
  --calibration-mode cross-group --partition-mode contiguous `
  --route-source router --hard-route-scale 6 --steps 0 `
  --hard-train-steps 0 --train-batches 1 --eval-batches 4 `
  --batch-size 8 --sequence-length 128 --alphas 1 0 `
  --calibration-text-file data/qwen_calibration.txt `
  --eval-text-file data/qwen_eval.txt --timing-warmup 3 `
  --timing-iterations 20 --dispatch-mode packed --seed 2026 `
  --output results/runs/qwen_dispatch_8layers_k6_packed_seed2026.json
```

The grouped and token-loop runs use the same command with only
`--dispatch-mode grouped` or `--dispatch-mode token-loop`.

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `tests/test_qwen_packed_dispatch.py`
- `results/runs/qwen_dispatch_8layers_k6_grouped_seed2026.json`
- `results/runs/qwen_dispatch_8layers_k6_token_loop_seed2026.json`
- `results/runs/qwen_dispatch_8layers_k6_packed_seed2026.json`
