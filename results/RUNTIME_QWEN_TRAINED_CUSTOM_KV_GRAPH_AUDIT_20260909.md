# V0.197 — Trained Qwen custom fixed-KV CUDA Graph audit

**Date:** 2026-09-09  
**Status:** `POSITIVE OPT-IN; QUALITY AND REPLAY PARITY PASS`  
**Branch:** `exp/track-runtime`

## Question

The V0.196 custom fixed-position KV cache replayed a sparse Qwen decode
smoke, but its children were not trained. This audit checks whether the same
runtime path works with the accepted trained K=5 quality recipe.

## Controlled recipe

- model: local `Qwen/Qwen3-0.6B`, float32 CUDA;
- replaced layers: `19,20,21,22,23,24,25,26`;
- 8 copied groups, 5 active groups (`62.5%` active);
- contiguous partition, rank-64 cross-group correction;
- `subset-router` with `subset-soft` target, temperature `0.25`;
- 100 router steps, 300 soft child steps, 300 hard child steps;
- hard route scale `5.0`, hard-phase learning rate `3e-4`;
- calibration: `data/qwen_calibration.txt`, batch 8 × sequence 128,
  8 batches; held-out evaluation: `data/qwen_eval.txt`, 4 batches.

The child training and sequential cascade order match the accepted V0.193
K=5 protocol. The benchmark then restores the dense parent for the baseline,
and measures the trained sparse cascade using the custom fixed-KV cache.

## Quality result

| metric | result |
|---|---:|
| teacher CE | `4.785308` |
| trained sparse CE | `4.821836` |
| CE delta | `+0.036528` |
| quality gate (`<= +0.05`) | **pass** |
| mean top-1 agreement to teacher logits | `80.47%` |

The result is consistent with the previous two-seed K=5 pass (`+0.03881`
and `+0.04036`) and does not show a quality regression from using the
trained child in the runtime audit. This is one fresh seed, so it extends
the existing quality evidence rather than replacing the two-seed claim.

The largest held-out local subset regrets were at layers 21 and 22:
`0.1484`/`0.1728` mean regret and `0.6205`/`0.5393` p95. This confirms that
the runtime result does not solve the remaining learned-routing problem at
lower active budgets; it only verifies that the trained child can enter the
runtime path correctly.

## Trained decode runtime and parity

The shape was batch 1, one decode token after a four-token prefix. The graph
used the custom fixed-position KV cache and the trained sparse children.

| path | mean latency |
|---|---:|
| dense parent eager | `26.653 ms` |
| trained sparse eager | `32.175 ms` |
| trained sparse CUDA Graph | `17.550 ms` |

- graph / dense parent: `0.658x`;
- graph / sparse eager: `0.545x`;
- replay-vs-eager max logit error: `6.68e-6`;
- alternate-token graph-vs-eager max logit error: `7.63e-6`;
- status: `PARITY_PASS` at the `1e-3` threshold.

The trained graph is therefore faster than the dense parent in this fixed
decode shape, while preserving the sparse eager result to float32 numerical
noise. This is a meaningful runtime result, but it is not yet a full
throughput or product-serving claim.

## Implementation note

The first attempt exposed a wrapper bug: `single_token_fast_path` was set on
the rank-64 correction wrapper instead of its nested
`TransferredRoutedQwenChild`. CUDA Graph capture consequently reached
`torch.bincount`, which is not permitted during capture. Propagating the flag
to the routed base fixed the issue; no routing formula or model weights were
changed.

## Decision

`V0.197` is **accepted as an opt-in runtime path**. It closes the specific
question “can an accepted trained K=5 child use the custom fixed-KV graph?”
The default model remains unchanged.

Still open:

- a true prefill-to-decode integration over multiple sequence lengths;
- `generate()` integration without manually managing the custom cache;
- dynamic-shape/shape-cache policy;
- a second trained seed for the graph runtime point;
- production kernel/stream safety and longer timing runs.

## Reproduction

```text
python -u benchmark_qwen_trained_graph_audit.py --local-files-only --device cuda
```

## Artifacts

- code: `benchmark_qwen_trained_graph_audit.py`;
- raw result: `results/runs/qwen_v0197_trained_custom_kv_graph_k5_seed2026.json`;
- custom cache implementation: `benchmark_qwen_custom_kv_graph.py`.
