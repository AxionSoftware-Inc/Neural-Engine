# V0.229 — Trained rank-1 correction long-budget audit

**Date:** 2026-09-09  
**Status:** `ACCEPTED OPT-IN CANDIDATE; DENSE GAP OPEN`  
**Branch:** `exp/track-runtime`

## Question

Earlier rank sweeps showed that a smaller low-rank correction can preserve the
trained K=5 quality gate while reducing the selected-FFN work. This audit asks
whether the smallest tested non-zero rank (`rank=1`) remains stable with the
longer `600/600/200` training recipe and repeated graph timing on two seeds.
The model, route policy and grouped dispatch implementation are unchanged;
only correction rank and the timing budget are under test.

## Protocol

- Teacher: `Qwen/Qwen3-0.6B`, layers 19–26, float32.
- Sparse recipe: `E=8`, `K=5`, `subset-router`, grouped selected-FFN dispatch.
- Calibration rank: `1`.
- Training: child/hard/router steps `600/600/200`.
- Timing: B8, prefix length 4, 8 CUDA-Graph warmups and 20 measured
  iterations.
- Seeds: `2026` and `17`.
- Quality gate: sparse CE delta must stay at or below `+0.05` versus the
  dense parent. Generation must produce exact graph/eager tokens.

## Results

| Seed | Sparse CE delta | Top-1 agreement | Dense graph | Grouped graph | Grouped-fused graph | Fused/dense | Fused/grouped |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026 | `+0.045878` | `79.35%` | `16.6752 ms` | `17.4838 ms` | `17.4837 ms` | `1.048x` | `1.000x` |
| 17 | `+0.035236` | `78.69%` | `16.7294 ms` | `17.5154 ms` | `17.4706 ms` | `1.044x` | `0.997x` |

Two-seed mean CE delta is `+0.040557`, and mean top-1 agreement is
`79.01%`. Both quality results pass the `+0.05` gate. The grouped-fused
variant is effectively neutral versus ordinary grouped at this shape; its
small seed-17 improvement does not justify a new automatic policy.

Both runs passed graph/eager parity and exact eight-token greedy-generation
parity (`grouped_vs_grouped_fused_exact_token_match=true`). Maximum final-logit
differences versus the single-token reference were `1.00e-5` and `1.10e-5`,
within the existing parity tolerance.

## Decision

- `rank=1` is the smallest tested non-zero correction that passes this
  two-seed long-budget quality check.
- Keep it as an **opt-in runtime candidate**; do not replace the rank-64
  compatibility default yet.
- The sparse path is still `1.044–1.048x` the dense parent at B8 graph time,
  so the dense-serving performance problem is not solved.
- More router or correction-rank shrinking is not the next priority. The next
  runtime target is selected-FFN launch/packing overhead: static-index or
  fully fused grouped dispatch that can beat the dense parent without changing
  route decisions.

## Reproduction

```text
python -u benchmark_qwen_trained_dispatch_path_audit.py --experiment V0.229_trained_rank1_long_audit --warmup 8 --iterations 20 --batch-sizes 8 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 2026 --output results/runs/v0_229_rank1_long_grouped_graph_audit.json
python -u benchmark_qwen_trained_dispatch_path_audit.py --experiment V0.229_trained_rank1_long_audit --warmup 8 --iterations 20 --batch-sizes 8 --prefix-lengths 4 --calibration-rank 1 --child-steps 600 --hard-steps 600 --router-steps 200 --seed 17 --output results/runs/v0_229_rank1_long_grouped_graph_audit_seed17.json
```

## Artifacts

- `benchmark_qwen_trained_dispatch_path_audit.py`
- `results/runs/v0_229_rank1_long_grouped_graph_audit.json`
- `results/runs/v0_229_rank1_long_grouped_graph_audit_seed17.json`
- `results/RUNTIME_QWEN_GROUPED_GRAPH_SAFE_AUDIT_20260909.md`
