# V0.110 Qwen Gated Attention-Free Child

## Question

Is the two-layer handoff failure mainly caused by using the wrong local
nonlinearity, or by insufficient child capacity?

## Protocol

The V0.109 two-layer protocol is held fixed: Qwen3-0.6B, late layers 25 and
26, eight train batches, four held-out batches, 300 local distillation steps
per child, and full alpha sweeps. The child is changed from

`LayerNorm -> Linear -> GELU -> Linear`

to an attention-free Qwen-like gated block

`Linear_gate -> SiLU * Linear_value -> Linear_out`.

Both width 384 and width 768 are tested. No attention or Qwen parent weights
are used in the child path at alpha=0.

```powershell
python -u benchmark_qwen_two_layer_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --first-layer 25 --second-layer 26 --inner-size 384 --child-kind swiglu --batch-size 4 --sequence-length 64 --train-batches 8 --eval-batches 4 --steps 300 --learning-rate 0.003 --seed 2026 --output results/runs/qwen_two_layer_swiglu_seed2026.json
python -u benchmark_qwen_two_layer_transplant.py --model Qwen/Qwen3-0.6B --local-files-only --device cuda --dtype float32 --first-layer 25 --second-layer 26 --inner-size 768 --child-kind swiglu --batch-size 4 --sequence-length 64 --train-batches 8 --eval-batches 4 --steps 300 --learning-rate 0.003 --seed 2026 --output results/runs/qwen_two_layer_swiglu_width768_seed2026.json
```

## Results

| child | scalar parameters each | alpha=0 two-child CE delta | teacher top-1 agreement | decision |
|---|---:|---:|---:|:---:|
| GELU, width 384 | 789,888 | +0.2377 | 91.50% | fail |
| SwiGLU, width 384 | 1,181,440 | +0.2230 | 91.21% | fail |
| SwiGLU, width 768 | 2,361,856 | +0.2277 | 92.68% | fail |

The gated form gives a small absolute improvement of 0.0147 CE at width 384,
but widening it by 2x does not improve the full handoff. One-child controls
remain good (the width-768 run gives +0.0343 for layer 25 only and +0.0071 for
layer 26 only). The failure therefore appears only when the two approximated
representations compose.

## Decision

**Changing the child activation and increasing its width are insufficient.**
The two-layer full-handoff gate remains rejected. This closes the simple
“more child capacity” and “copy Qwen's gated activation” repairs for the
current local-regression protocol.

The remaining plausible issue is the interface between successive child
blocks: a small hidden-state error is amplified by the next layer and by the
unchanged residual stream. The next architecture test should add a compact
learned interface/state-calibration path that is trained jointly, then measure
whether it reduces error without restoring parent FFN or attention parameters.

## Artifacts

- `benchmark_qwen_two_layer_transplant.py`
- `results/runs/qwen_two_layer_swiglu_seed2026.json`
- `results/runs/qwen_two_layer_swiglu_width768_seed2026.json`
