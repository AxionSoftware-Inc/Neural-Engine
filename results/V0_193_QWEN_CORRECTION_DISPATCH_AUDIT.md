# V0.193 — Qwen correction dispatch memory/runtime audit

**Date:** 2026-09-07  
**Status:** `POSITIVE ENGINEERING RESULT; QUALITY-PARITY PASS`  
**Branch:** `exp/test-p003-progressive-capacity`

## Hypothesis

The rank-64 cross-group correction was mathematically correct but its hard
path indexed `mix_in[selected]` and `mix_out[selected]`. At Qwen width this
materialized a `[tokens, K, rank, hidden]` tensor per layer. For
`tokens=1024`, `K=6`, `rank=64`, and `hidden=1024`, that is about 402 million
float32 entries (roughly 1.6 GB) before the two projections.

The patch keeps the correction formula unchanged. It flattens selected tokens,
processes only the selected rows for each expert, and accumulates the result
with `index_add_`. Empty experts are skipped. This removes the large indexed
weight gather; it does not force a different route or active budget.

## Correctness control

`tests/test_qwen_packed_dispatch.py` now compares the new hard path with the
original two-einsum reference formula on a deterministic CPU fixture. The
comparison passes at `atol=rtol=1e-6`.

The test suite also retains the grouped and grouped-fused dispatch parity
controls.

## Quality benchmark

The benchmark exactly reuses the V0.174 eight-layer K=6 recipe:

- `Qwen/Qwen3-0.6B`, layers `19–26`;
- E=8 groups, K=6 active, contiguous partition;
- rank-64 cross-group correction;
- `subset-router`, `subset-soft`, temperature `0.25`;
- 300 distillation steps, 300 hard-training steps at LR `3e-4`;
- 100 router-supervision steps;
- batch 8 × sequence length 128, 8 training batches, 4 evaluation batches;
- float32 CUDA, shared alpha=0 quality gate.

The exact commands were run with `--dispatch-mode grouped`, changing only the
correction implementation under test:

```text
python -u benchmark_qwen_multi_layer_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --layers 19,20,21,22,23,24,25,26 --child-kind qwen-transfer-sparse --num-experts 8 --active-experts 6 --calibration-rank 64 --calibration-mode cross-group --partition-mode contiguous --route-source subset-router --hard-route-scale 6 --router-target subset-soft --router-target-temperature 0.25 --steps 300 --hard-train-steps 300 --hard-learning-rate 3e-4 --router-supervision-steps 100 --train-batches 8 --eval-batches 4 --batch-size 8 --sequence-length 128 --alphas 1 0 --calibration-text-file data/qwen_calibration.txt --eval-text-file data/qwen_eval.txt --timing-warmup 1 --timing-iterations 2 --dispatch-mode grouped --seed 2026 --output results/runs/qwen_v0193_8layers_k6_grouped_rank64_optcorr_seed2026.json

python -u benchmark_qwen_multi_layer_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --layers 19,20,21,22,23,24,25,26 --child-kind qwen-transfer-sparse --num-experts 8 --active-experts 6 --calibration-rank 64 --calibration-mode cross-group --partition-mode contiguous --route-source subset-router --hard-route-scale 6 --router-target subset-soft --router-target-temperature 0.25 --steps 300 --hard-train-steps 300 --hard-learning-rate 3e-4 --router-supervision-steps 100 --train-batches 8 --eval-batches 4 --batch-size 8 --sequence-length 128 --alphas 1 0 --calibration-text-file data/qwen_calibration.txt --eval-text-file data/qwen_eval.txt --timing-warmup 1 --timing-iterations 2 --dispatch-mode grouped --seed 2027 --output results/runs/qwen_v0193_8layers_k6_grouped_rank64_optcorr_seed2027.json
```

| seed | teacher CE | sparse alpha=0 CE delta | quality gate |
|---:|---:|---:|:---:|
| 2026 | `4.785308` | `+0.011406` | pass |
| 2027 | `4.785308` | `+0.018373` | pass |

Both results are below the `+0.05` gate. The corresponding V0.174 reference
was `+0.01103` and `+0.02117`; therefore the implementation change does not
produce a quality regression in this two-seed control. This is a dispatch
parity result, not evidence that the router or capacity problem is solved.

## Runtime result

The previous grouped rank-64 implementation measured `506.60 ms / 237.27 ms`
for sparse/dense, or `2.135x`, in the comparable eight-layer smoke. The
optimized correction smoke measured `268.42 ms / 237.94 ms`, or `1.128x`.
That is approximately 47% less sparse time and 2.6x lower sparse latency than
the old rank-64 correction path.

The trained two-seed runs measured:

| seed | dense mean ms | sparse mean ms | sparse/dense |
|---:|---:|---:|---:|
| 2026 | `273.564` | `271.146` | `0.991x` |
| 2027 | `271.693` | `272.803` | `1.004x` |

The two-iteration timing has normal CUDA noise, so the stable claim is that
the former 2.1x dispatch penalty was removed in this protocol; a production
throughput claim needs a longer timing run.

## K=5 active-budget validation

Because the earlier V0.174 K=5 scale-normalized control passed quality at a
lower active budget but remained `1.91x` slower, the same patch was checked at
E=8/K=5 with `hard_route_scale=5`, `timing-warmup=2`, and five timing
iterations.

| seed | sparse alpha=0 CE delta | old timing / parent | new timing / parent |
|---:|---:|---:|---:|
| 2026 | `+0.038812` | `1.91x` | `1.134x` |
| 2027 | `+0.040363` | `1.91x` | `1.129x` |

Both K=5 seeds pass the `+0.05` quality gate. The new sparse means were
`272.321 ms` vs `240.073 ms` dense (seed 2026) and `271.095 ms` vs
`240.157 ms` dense (seed 2027). This establishes K=5/62.5% active as the
current best lower-budget operating point: it preserves the existing quality
result while reducing the old dispatch penalty to roughly 13% overhead.

The K=5 commands are identical to the K=6 commands above except for
`--active-experts 5`, `--hard-route-scale 5`, the seed, and the output path.
The JSON artifacts are listed below.

## Decision and next step

`V0.193` is accepted as the new grouped correction implementation because it
preserves the formula and passes both quality seeds while removing the large
memory gather. This closes the immediate rank-64 correction-dispatch
bottleneck, but P-006 instrumentation and the larger K=4 router-gap/P-001/
P-002 research problems remain open. K=5 is the preferred lower-budget
operating point; K=6 remains the higher-margin quality reference.

Do not interpret this as proof that the attention-free model beats Qwen. It is
a systems result: the existing sparse child can now run near dense latency in
the tested 8-layer K=6 configuration.

## Artifacts

- Code: `benchmark_qwen_multi_layer_transplant.py`
- Parity test: `tests/test_qwen_packed_dispatch.py`
- Seed 2026 JSON: `results/runs/qwen_v0193_8layers_k6_grouped_rank64_optcorr_seed2026.json`
- Seed 2027 JSON: `results/runs/qwen_v0193_8layers_k6_grouped_rank64_optcorr_seed2027.json`
- K=5 seed 2026 JSON: `results/runs/qwen_v0193_8layers_k5_grouped_rank64_optcorr_seed2026.json`
- K=5 seed 2027 JSON: `results/runs/qwen_v0193_8layers_k5_grouped_rank64_optcorr_seed2027.json`
