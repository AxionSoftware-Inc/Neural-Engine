# V0.190 Qwen FP16 selected-dispatch audit

## Question

Can the selected Qwen groups use FP16 Tensor Core GEMMs to recover enough
latency to make the sparse path practical, while keeping the model and router
unchanged?

## Protocol

The V0.188 matched runtime setup was used: Qwen3-0.6B, eight layers (19--26),
E=8/K=6 contiguous groups, rank-64 cross-group correction, batch 8, sequence
128, float32 parent model, and the same held-out text. Only selected-group
GEMMs use cached FP16 weights and FP16 inputs; their output is accumulated back
in float32. The mode is opt-in as `--dispatch-mode packed-fp16`.

## Results

### One-layer timing

| Backend | Parent ms/batch | Sparse ms/batch | Sparse / parent |
|---|---:|---:|---:|
| packed float32 | 236.89 | 269.78 | 1.139x |
| packed-fp16 | 236.65 | 268.25 | 1.134x |

This is only about 0.6% faster than the packed float32 control.

### Eight-layer timing and quality smoke

With 5 timing iterations, FP16 sparse timing was `494.61 ms` versus
`238.62 ms` parent, a `2.073x` ratio. The comparable packed-fused control was
`509.91 ms` (`2.121x`); therefore FP16 saves about 3% relative to that control,
but it is still slower than dense inference.

The same short eight-layer evaluator produced alpha=0 CE delta `+0.0864`, so
it failed the existing `<=+0.05` quality gate. This is a backend control, not
a new routing or capacity result.

## Decision

**Rejected for adoption.** FP16 reduces the dispatch cost slightly but does
not create a speedup and introduces a measurable quality regression in this
control. The default remains grouped float32. A worthwhile speed result now
requires a genuinely tiled/grouped GEMM backend with proper expert batching,
not only lower precision.

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
  --timing-iterations 5 --dispatch-mode packed-fp16 --seed 2026 `
  --output results/runs/qwen_dispatch_8layers_k6_packed_fp16_seed2026.json
```
