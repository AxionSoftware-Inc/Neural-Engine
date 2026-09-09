# V0.242 — Folded correction grouped-dispatch audit

**Date:** 2026-09-09  
**Status:** `PARITY PASS; SMALL OPT-IN SPEEDUP; DEFAULT UNCHANGED`  
**Branch:** `exp/track-runtime`

## Hypothesis

The selected-output correction has the form

```text
v_corrected = v + mix_out[e] @ (mix_in[e] @ v)
```

Because the Qwen output projection is linear, this can be folded once into
an effective per-expert output weight:

```text
W_effective[e] = W_out[e] + mix_out[e] @ mix_in[e] @ W_out[e]
```

The grouped path can then avoid the per-forward low-rank correction
contractions. The route IDs, selected K, accumulation scale, and copied
expert weights remain unchanged.

## Implementation note

The first probe exposed an index-label bug in the initial einsum: output
hidden and input hidden had been given the same label. That version produced a
large logit mismatch and was not accepted. The contraction was corrected to
use distinct output/input hidden indices (`eor,eri,eig->eog`), then rechecked
with non-zero correction weights on both a real Qwen layer and the unit test.
Only the corrected results below are valid.

## Protocol

Local `Qwen/Qwen3-0.6B`, layers `19–26`, `E=8`, `K=5`, rank-1 correction,
float32 CUDA, and the `600/600/200` trained-cascade recipe. Prefix length was
4; batch sizes were 1, 8, and 32. Each path used 10 CUDA-Graph warmups and
20 timing iterations. The baseline was `grouped-adaptive`; the probe used
`grouped-adaptive-effective-output` with the same correction parameters.

## Results

Graph latency is in milliseconds. Ratio is `effective-output / adaptive`.

| seed | batch | adaptive | folded output | ratio | change |
|---:|---:|---:|---:|---:|---:|
| 2026 | 1 | 11.162 | 11.128 | 0.9969 | −0.31% |
| 2026 | 8 | 17.033 | 16.904 | 0.9924 | −0.76% |
| 2026 | 32 | 34.599 | 34.407 | 0.9945 | −0.55% |
| 17 | 1 | 11.203 | 11.098 | 0.9906 | −0.94% |
| 17 | 8 | 17.002 | 16.826 | 0.9896 | −1.04% |
| 17 | 32 | 34.465 | 34.333 | 0.9962 | −0.38% |

Both seeds passed CUDA Graph/eager parity; final-logit differences versus the
single-token reference remained in the existing float32 range (`<=1.4e-5`),
and eight-token greedy generation matched the grouped baseline exactly. The
quality measurements stayed within the same trained recipe (`CE delta
`+0.04648` for seed 2026 and `+0.03628` for seed 17`); this runtime fold does
not claim a quality improvement.

## Decision

- Keep the folded-output path as a parity-safe opt-in micro-optimization.
- Do not make it the global default: the gain is below 1.1% and does not close
  the dense gap.
- V0.240 remains the bottleneck diagnosis: `pack` and `select/correction`
  dominate. Folding correction removes only part of the latter.
- The next material runtime target is one route-aware compiled kernel that
  combines selected-row packing, output projection/correction, and token
  accumulation without intermediate expert-major workspace.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --warmup 10 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-effective-output --experiment V0.242_grouped_effective_output_probe --output results/runs/v0_242_grouped_effective_output_seed2026.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --warmup 10 --iterations 20 --batch-sizes 1 8 32 --prefix-lengths 4 --include-grouped-adaptive --include-grouped-adaptive-effective-output --experiment V0.242_grouped_effective_output_probe --output results/runs/v0_242_grouped_effective_output_seed17.json
```

## Artifacts

- `benchmark_qwen_multi_layer_transplant.py`
- `benchmark_qwen_trained_dispatch_path_audit.py`
- `tests/test_qwen_packed_dispatch.py`
- `results/runs/v0_242_grouped_effective_output_seed2026.json`
- `results/runs/v0_242_grouped_effective_output_seed17.json`
