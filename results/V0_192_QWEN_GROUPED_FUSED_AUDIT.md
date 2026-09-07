# V0.192 Qwen grouped-fused dispatch audit

## Question

Does combining the gate and value projections into one batched GEMM reduce the
cost of the selected Qwen bank without changing routing or circuit outputs?

## Protocol

Qwen3-0.6B, float32 CUDA, contiguous E=8/K=6 groups and rank-64 cross-group
correction were kept. `grouped-fused` uses the same metadata, packing and
scatter code as `grouped`; only the two first projection GEMMs are concatenated
into one batched GEMM. It is opt-in and the default remains `grouped`.

## Results

Isolated layer-26 MLP profile, batch 8 × sequence 128:

| Backend | CUDA time | Relative to dense parent |
|---|---:|---:|
| dense parent | `3.935 ms` | `1.00x` |
| grouped | `7.564 ms` | `1.92x` |
| grouped-fused | `6.549 ms` | `1.66x` |

The isolated MLP improves by about 13.4%, mainly because three batched
projection launches become two. However, end-to-end timing does not preserve
that gain. The 8-layer Qwen smoke (layers 19--26, 5 timing iterations) was:

| Backend | Parent ms/batch | Sparse ms/batch | Ratio |
|---|---:|---:|---:|
| grouped-fused | `238.88` | `506.93` | `2.122x` |

This is indistinguishable from the V0.188 grouped reference (`506.60 ms`,
`2.135x`) under ordinary timing noise. The short alpha=0 CE delta was
`+0.08643`, matching the untrained sparse-bank control; this is not a new
quality claim and does not pass the quality gate.

The grouped-fused path matches grouped output on the CPU unit test at
`1e-6` tolerance.

## Decision

**Rejected for adoption.** Projection fusion helps the isolated child but does
not create an end-to-end speedup. The remaining cost is the combination of
small expert GEMMs, ragged token grouping and scatter/accumulation. A real
runtime candidate needs a fused tiled grouped kernel that owns all three
stages; another launch-level PyTorch rearrangement is not enough.

No router, circuit-bank or model-capacity conclusion follows from V0.192.
P-006 remains active and 700M/1B scaling remains deferred.

## Reproduction

```powershell
python -u benchmark_qwen_dispatch_profile.py --dispatch-mode grouped
python -u benchmark_qwen_dispatch_profile.py --dispatch-mode grouped-fused

python -u benchmark_qwen_multi_layer_transplant.py `
  --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 `
  --layers 19,20,21,22,23,24,25,26 --child-kind qwen-transfer-sparse `
  --num-experts 8 --active-experts 6 --calibration-rank 64 `
  --calibration-mode cross-group --partition-mode contiguous `
  --route-source router --hard-route-scale 6 --steps 0 `
  --train-batches 1 --eval-batches 4 --batch-size 8 --sequence-length 128 `
  --alphas 1 0 --calibration-text-file data/qwen_calibration.txt `
  --eval-text-file data/qwen_eval.txt --timing-warmup 2 `
  --timing-iterations 5 --dispatch-mode grouped-fused --seed 2026
```
