# V0.209 — Rank-4 correction audit

**Date:** 2026-09-09  
**Status:** `RANK-4 ACCEPTED OPT-IN; DEFAULT 64 PRESERVED`  
**Branch:** `exp/track-runtime`

## Question

Rank 8 was the smallest accepted correction configuration in the previous
audit. This follow-up tests whether the correction can be compressed further
without losing the K=5 quality gate. The router, eight replaced layers,
training data, training steps, graph path, and evaluation protocol are kept
unchanged; only `calibration_rank=4` changes.

## Results

| seed | CE delta | top-1 agreement | B8 parent | B8 sparse eager | B8 sparse graph | graph / dense | graph / sparse eager | max parity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | `+0.037330` | `0.7947` | `25.838 ms` | `39.022 ms` | `34.597 ms` | `1.339x` | `0.887x` | `1.05e-5` |
| 17 | `+0.040037` | `0.7871` | `26.500 ms` | `38.044 ms` | `35.258 ms` | `1.330x` | `0.927x` | `8.58e-6` |
| 42 | `+0.040855` | `0.7939` | `25.576 ms` | `38.008 ms` | `34.186 ms` | `1.337x` | `0.899x` | `9.06e-6` |

The quality gate is `CE delta < +0.05`; all three independent seeds pass it.
Graph replay remains within the existing `1e-3` parity tolerance, and both
trained generation checks pass exactly: graph/eager token equality and reused
shape equality. The B8 graph is still slower than the dense parent, so this is
not yet a dense-latency breakthrough; it is a lower-capacity correction result.

## Comparison and decision

| correction rank | tested seeds | CE delta range | B8 graph / dense | decision |
|---:|---:|---:|---:|---|
| 64 | 2026 | `+0.035745` | `1.491x` | compatibility default |
| 32 | 2026, 17 | `+0.031617…+0.039860` | `1.411x/1.344x` | accepted opt-in |
| 16 | 2026 | `+0.036007` | `1.479x` | not preferred |
| 8 | 2026, 17 | `+0.024122…+0.029971` | `1.390x/1.008x` | accepted opt-in |
| 4 | 2026, 17, 42 | `+0.037330…+0.040855` | `1.339x/1.330x/1.337x` | smallest tested viable opt-in |
| 0 | 2026 | `+0.091023` | `1.047x` | quality rejected |

Rank 4 is therefore accepted as an opt-in candidate, not promoted to the
default. Rank 0 is the control showing that removing correction entirely gives
near-dense runtime but unacceptable quality. Rank 4 now has three-seed support,
but still needs a longer training-budget check before any default change. The next
implementation target remains a static-index/fused correction kernel; lowering
rank alone does not remove the dispatch overhead.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --seed 2026 --warmup 40 --iterations 100 --correction-backend-iterations 15 --single-token-backend-iterations 30 --compiled-child-iterations 2 --batch-sizes 1 8 --output results/runs/v0_209_rank4_seed2026.json
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --seed 17 --warmup 40 --iterations 100 --correction-backend-iterations 15 --single-token-backend-iterations 30 --compiled-child-iterations 2 --batch-sizes 1 8 --output results/runs/v0_209_rank4_seed17.json
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda --calibration-rank 4 --seed 42 --warmup 40 --iterations 100 --correction-backend-iterations 15 --single-token-backend-iterations 30 --compiled-child-iterations 2 --batch-sizes 1 8 --output results/runs/v0_209_rank4_seed42.json
```

## Artifacts

- parameterized audit: `benchmark_qwen_trained_graph_audit.py`;
- correction implementation and rank control:
  `benchmark_qwen_multi_layer_transplant.py`;
- prior rank controls:
  `RUNTIME_QWEN_CORRECTION_RANK8_AUDIT_20260909.md` and
  `RUNTIME_QWEN_CORRECTION_RANK_AUDIT_20260909.md`;
- rank-zero quality control: `RUNTIME_QWEN_CORRECTION_ABLATION_20260909.md`.
