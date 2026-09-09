# Runtime — Qwen custom fixed-KV CUDA Graph decode

**Sana:** 2026-09-09  
**Branch:** `exp/track-runtime`  
**Qaror:** `POSITIVE OPT-IN`; trained quality, prefill integration and shape
cache remain open.

## Muammo

The generic Transformers `StaticCache` is not safe to replay in a CUDA Graph
on this environment: its `StaticLayer.update()` advances an internal
`cumulative_length` on every replay. The earlier diagnostic therefore had
large logit errors even for the dense parent. A decode graph needs a KV buffer
that writes the current token to a fixed position without advancing Python or
tensor state during replay.

## Yechim

`FixedDecodeLayer` subclasses the Transformers static layer and keeps its
preallocated KV tensors. During prefix prefill it uses the normal cumulative
position. Before graph capture, each layer receives a fixed one-token decode
position; subsequent graph replays overwrite that KV slot and leave the
sequence length unchanged. This keeps the Qwen model API at
`past_key_values=...`, `use_cache=True` while removing the unsafe counter
mutation.

The Qwen model, router, copied groups, active budget and FFN formulas are not
changed. The runtime smoke uses Qwen3-0.6B, float32, RTX 3060, prefix length 4,
one decode token, and eight replaced layers `[0,4,8,12,16,20,24,26]`.

## Natijalar

| Active groups | Iterations | Dense parent | Sparse eager | Custom-KV graph | Graph / parent | Graph / eager | Replay max error | Alternate-token max error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K=5/8 | 50 | `28.833 ms` | `38.152 ms` | `18.635 ms` | `0.646x` | `0.488x` | `1.11e-5` | `7.99e-6` |
| K=5/8 | 200 | `28.491 ms` | `29.421 ms` | `14.703 ms` | `0.516x` | `0.500x` | `8.58e-6` | `9.66e-6` |
| K=6/8 | 100 | `28.720 ms` | `29.939 ms` | `17.084 ms` | `0.595x` | `0.571x` | `1.22e-5` | `8.58e-6` |

K=5 is the current lower-budget quality operating point: the separate
two-seed Qwen audit passed its `+0.05` CE gate at `+0.03881/+0.04036`; K=6 is
the higher-margin reference. This report does not repeat that quality audit:
the child routers here are runtime smoke modules, not trained checkpoint
artifacts.

## Input and cache correctness

The graph was captured with one token, then its GPU input buffer was replaced
with a different token and replayed. The alternate replay matched an eager
forward from the same prefix cache within `9.66e-6` at K=5 and `8.58e-6` at
K=6. The normal replay matched eager within `1.22e-5` in the worst listed
run. The earlier generic `StaticCache` failure is not present with the custom
fixed-position layer.

## Talqin va cheklovlar

This is the first `use_cache=True` graph path with numerical parity in the
current Qwen runtime. K=5 graph replay is approximately 49%--50% faster than
the corresponding eager sparse path and approximately 48%--52% of the dense
parent latency in these repeats. The gain is a launch-overhead result, not a
new model-quality or attention-removal result.

The fixed-position adapter currently represents one decode position per
captured graph. A production wrapper must maintain a graph per supported
shape/position policy or use a custom dynamic KV update kernel. Prefill, long
sequences, batch >1, trained K=5/K=6 children, and `generate()` integration
remain separate gates.

## Qaror va keyingi qadam

- Keep custom fixed-KV graph replay as an opt-in runtime candidate.
- Do not replace the default eager path yet.
- Next: run the same graph wrapper around a trained K=5 quality checkpoint,
  then add a small shape cache and a real prefill-to-decode caller.
- Generic `StaticCache` graph replay remains rejected; its diagnostic is kept
  in `results/RUNTIME_QWEN_CUDA_GRAPH.md`.

## Reproduksiya

```powershell
python -u benchmark_qwen_custom_kv_graph.py `
  --layers 0,4,8,12,16,20,24,26 --active-experts 5 `
  --iterations 200 --warmup 50

python -u benchmark_qwen_custom_kv_graph.py `
  --layers 0,4,8,12,16,20,24,26 --active-experts 6 `
  --iterations 100 --warmup 30
```

## Artifact

- `benchmark_qwen_custom_kv_graph.py`
- Quality reference: `results/V0_193_QWEN_CORRECTION_DISPATCH_AUDIT.md`
